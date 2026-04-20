# Sandbox 最小修复方案

## 目标

这份方案只解决当前最可能导致评测 nondeterminism 的两个根因：

1. `stdout/stderr` 读取窗口过短，导致空输出或截断输出。
2. 进程清理时机过早，在完整 drain pipe 之前就 kill process tree。

本方案刻意不做大重构，不改 verifier，不改 nginx/LB，只修 SandboxFusion server 端最小必要逻辑。

相关文件：

- [execution.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/execution.py)
- [base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py)
- [local.yaml](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/configs/local.yaml)

---

## 当前判断

### 主因 1：输出采集竞态

[execution.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/execution.py) 里当前：

```python
res = await asyncio.wait_for(fd.read(1024 * 1024), timeout=0.0001)
```

问题：

- `0.0001s` 的 timeout 极小。
- 在高并发下，即使子进程逻辑上已经结束，pipe 里的数据也可能还没被完整调度到 reader。
- 结果就是：
  - `stdout=""`
  - 或 `stdout` 少最后几行
  - 或 `stderr` 被截断

这和当前现场现象完全一致：

- 同一个 `response`
- 同一个 testcase
- 有时 `success`
- 有时 `wrong_answer`
- 常见表象是 `actual=''` 或 `actual` 缺尾部

### 次因 2：kill-before-drain

[base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py) 里当前：

1. `await p.wait()`
2. `finally` 中先 `kill_process_tree(p.pid)`
3. 之后再 `await get_output_non_blocking(p.stdout / p.stderr)`

这会放大上面的竞态：

- timeout 路径下尤其明显
- 正常结束路径也不稳，因为 `finally` 总会进入
- 如果子进程树还有残留，过早 kill 可能让尚未完全刷入 pipe 的输出丢失

### 非当前主因

[local.yaml](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/configs/local.yaml) 当前：

```yaml
sandbox:
  isolation: none
  cleanup_process: false
  restore_bash: false
```

所以：

- `cleanup_process()` 不是当前线上漂移的主因
- 但 `isolation: none` 仍然让高并发评测更脆弱，只是不是这次最先要修的地方

---

## 最小修复原则

这次只做 3 件事：

1. 去掉 `0.0001s` 的极短非阻塞读。
2. 正常完成路径改成“短暂 grace drain -> 如未收敛则清理 request process group -> guarded final drain”。
3. timeout/异常路径也统一成“先终止、再 drain、再返回”。

不做的事：

- 不改 verifier 聚合逻辑
- 不改 nginx LB
- 不改 dataset / eval 协议
- 不启用更重的隔离模式
- 不重构整个 runner 生命周期

---

## 建议改动

## 改动 1：替换 `get_output_non_blocking()`

文件：
- [execution.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/execution.py)

### 当前问题

当前函数语义是：

- 尝试快速读一点输出
- 超时就返回空串

这不适合“子进程已经结束之后的最终结果采集”。

### 最小修法

把它改成两个函数：

1. `drain_stream(fd)`  
   用于**最终收集**，在子进程已结束或已被终止后，完整读取 pipe。

2. `get_output_non_blocking(fd)`  
   如果还想保留，可仅用于某些真正的 probe 场景，但不要用于最终 verdict。

### 建议实现

```python
async def drain_stream(fd) -> str:
    if fd is None:
        return ""
    try:
        data = await fd.read()
    except Exception as e:
        return f"[ReadError] {e}"
    return try_decode(data)
```

这里有一个重要约束：

- **final verdict 路径不能把 drain 超时静默转换成 `\"\"`**

否则会把当前最想修掉的问题重新引回来：

- 空 stdout
- 被误判成 wrong_answer
- 但调用链表面上没有异常

### 推荐口径

对最终判题结果，**不能再用 `0.0001s` 这种极短 probe 读**，但也不能简单地假设：

- 进程一结束
- `await fd.read()`
- 就一定会很快拿到 EOF

因为在 `isolation: none` 下，如果用户程序 fork 出子进程并继承了 `stdout/stderr`，主进程退出后，pipe 仍可能被后代持有。  
这时“无超时完整 drain”本身就可能卡死。

所以更准确的要求是：

- 最终 verdict 路径**不能静默回空串**
- 但 final drain 也不能在没有 request-scoped cleanup 的前提下无限等待

### 推荐实现方向

保留一个“完整 drain”的基础实现：

```python
async def drain_stream(fd) -> str:
    if fd is None:
        return ""
    try:
        data = await fd.read()
    except Exception as e:
        return f"[ReadError] {e}"
    return try_decode(data)
```

但真正使用时要配合**请求级 process group/session 清理**，见下文。

如果后面一定要给 drain 加保护超时，也不能返回空串；至少要返回显式标记，例如：

```python
return "[DrainTimeout]"
```

或单独走一个 `read_status="drain_timeout"` 字段，让上层能看见这次输出采集不可信。

---

## 改动 2：`run_command_bare()` 改成 request-scoped grace drain + cleanup + guarded final drain

文件：
- [base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py)

### 当前问题

当前结构里：

- `finally` 总会执行
- `finally` 里可能先 kill process tree
- 之后才读 `stdout/stderr`

这会让输出采集和进程终止互相竞态。

### 最小修法

把逻辑明确拆成两条路径，并补一条**基于 request-scoped process group 的 cleanup**：

1. **正常完成路径**
   - `await p.wait()`
   - 先做一段短暂、可控的“grace drain / grace period”
   - 如果 pipe 未自然收敛，则清理该请求的残留 process group
   - 再做带显式状态的 final drain
   - 返回 `Finished`

2. **超时路径**
   - 先 `kill_process_tree`
   - 再 `await p.wait()` 或给一个短等待
   - 再做带显式状态的 final drain
   - 返回 `TimeLimitExceeded`

### 为什么正常完成后不能直接无超时 drain

如果用户程序 fork 出后代并继承了 `stdout/stderr`：

- 主进程结束
- `p.wait()` 返回
- 但 pipe 仍未 EOF

这时如果直接：

- `await drain_stream(p.stdout)`

就可能一直等不到 EOF，把原来的 nondeterminism 换成 hang。

所以文档这里要明确：

- **正常路径不能简单写成 `wait -> 无超时 drain -> cleanup`**
- 必须先有“请求作用域内可追踪的 cleanup 手段”

### 为什么正常完成后仍要 cleanup

这一点不能完全省掉。

因为在当前：

- `isolation: none`
- 用户代码可能自己 fork 子进程
- 主进程退出不代表所有后代都退出

如果正常完成路径完全不做 cleanup，就可能把问题从：

- 输出采集竞态

换成：

- orphan subprocess 泄漏
- 后续请求互相干扰

所以更稳的做法不是“正常路径完全不清理”，也不是“退出后靠父 PID 反查 children”，而是：

- **在进程创建时就给每个请求分配独立 process group / session**
- 正常完成后先给一个短暂 grace period
- 如果 pipe 仍未收敛，就清理该 request 的残留 process group
- 然后再做最终 drain
- 不做全局 cleanup

### 建议新增 helper

推荐不要把 cleanup 建立在“父进程退出后再按 `psutil.Process(pid).children()` 反查”上。

原因：

- 父进程退出后，原 PID 可能已经消失
- 残留后代可能被 reparent 到 init/systemd
- 这时再按旧父 PID 反查 descendants，可能什么也找不到

所以这里更稳的最小方案是：

- **请求启动时就创建独立 process group / session**
- 后续 cleanup 直接按这个 request-scoped 标识去做

推荐新增的 helper 方向是：

```python
def kill_process_group(pgid: int):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warn(f"error on killing process group {pgid}: {e}")
```

如果不想直接 `SIGKILL`，也可以先 `SIGTERM` 再短等待后 `SIGKILL`。

### 进程创建侧要求

在 [base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py) 的 subprocess 创建时，需要保证每个请求有独立 process group / session。

可选方式：

- `start_new_session=True`
- 或在 `preexec_fn` 里做 `os.setsid()`

如果走 `preexec_fn` 这条路线，需要特别注意与现有 preexec 逻辑兼容：

- `os.setsid()` 必须与当前的内存限制、`set_uid` 等 preexec steps **组合执行**
- 不能为了加 process group / session，直接覆盖掉现有 `preexec_fn`
- 否则会把本来已有的 memory limit / uid 隔离悄悄弄丢

重点不是具体 API，而是：

- **后续 cleanup 必须依赖一个在进程退出后仍可追踪的 request scope**

### 原先的 `kill_child_processes(pid)` 方案为何不够稳

下面这种“退出后按原父 PID 找 descendants”的方式，只能作为弱备选，不适合当主方案：

```python
def kill_child_processes(pid: int):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception as e:
        logger.warn(f"error on killing child processes: {e}")
```

因为它依赖：

- 原父进程仍可被稳定索引
- descendants 仍保留在该 parent lineage 下

这两个前提在父进程已退出后都不稳。

### 建议伪代码

```python
timed_out = False
execution_time = None
pgid = ...  # request 启动时确定的 process-group id

try:
    await asyncio.wait_for(p.wait(), timeout=timeout)
    execution_time = time.time() - start_time
except asyncio.TimeoutError:
    timed_out = True
    execution_time = time.time() - start_time
    if psutil.pid_exists(p.pid):
        kill_process_group(pgid)
    try:
        await asyncio.wait_for(p.wait(), timeout=1.0)
    except Exception:
        pass

if not timed_out:
    try:
        stdout_text = await asyncio.wait_for(drain_stream(p.stdout), timeout=0.2)
        stderr_text = await asyncio.wait_for(drain_stream(p.stderr), timeout=0.2)
        stdout_status = "ok"
        stderr_status = "ok"
    except asyncio.TimeoutError:
        # 说明很可能仍有 descendants 持有 pipe
        kill_process_group(pgid)
        try:
            await asyncio.wait_for(p.wait(), timeout=1.0)
        except Exception:
            pass
        try:
            stdout_text = await asyncio.wait_for(drain_stream(p.stdout), timeout=1.0)
            stdout_status = "ok_after_cleanup"
        except asyncio.TimeoutError:
            stdout_text = "[DrainTimeout]"
            stdout_status = "drain_timeout"
        try:
            stderr_text = await asyncio.wait_for(drain_stream(p.stderr), timeout=1.0)
            stderr_status = "ok_after_cleanup"
        except asyncio.TimeoutError:
            stderr_text = "[DrainTimeout]"
            stderr_status = "drain_timeout"
else:
    try:
        stdout_text = await asyncio.wait_for(drain_stream(p.stdout), timeout=1.0)
        stdout_status = "ok_after_timeout_cleanup"
    except asyncio.TimeoutError:
        stdout_text = "[DrainTimeout]"
        stdout_status = "drain_timeout"
    try:
        stderr_text = await asyncio.wait_for(drain_stream(p.stderr), timeout=1.0)
        stderr_status = "ok_after_timeout_cleanup"
    except asyncio.TimeoutError:
        stderr_text = "[DrainTimeout]"
        stderr_status = "drain_timeout"

if config.sandbox.cleanup_process:
    cleanup_process()
if config.sandbox.restore_bash:
    ensure_bash_integrity()

if timed_out:
    return CommandRunResult(
        status=CommandRunStatus.TimeLimitExceeded,
        execution_time=execution_time,
        stdout=stdout_text,
        stderr=stderr_text,
    )

return CommandRunResult(
    status=CommandRunStatus.Finished,
    execution_time=execution_time,
    return_code=p.returncode,
    stdout=stdout_text,
    stderr=stderr_text,
)
```

### 关键点

- 不要在“正常完成路径”里无条件 kill process tree。
- 但也不要完全取消 cleanup。
- 正常完成路径不能假设无超时 drain 一定安全。
- cleanup 目标应该是：
  - **请求级 process group / session**
  - 而不是退出后再按父 PID 反查 children
- timeout/异常路径直接 kill request process group。
- 正常完成路径可先给短 grace drain；若 pipe 不收敛，再清理 request process group 后做 guarded final drain。
- cleanup 之后的 final drain 也必须保留最后一道防挂保护。
- final verdict 前统一 drain output，并且不能静默回空串；若最终仍读不到，必须显式标成 `"[DrainTimeout]"` 或 `read_status="drain_timeout"`。

---

## 改动 3：保留最小诊断信息

这不是必须项，但很建议一起做，成本很低。

在 `CommandRunResult` 或日志里补一条轻量诊断：

- `timed_out`
- `stdout_len`
- `stderr_len`
- `return_code`

目的：

- patch 后如果还有漂移，能快速确认是否仍然是“空 stdout/短 stdout”

如果不想改返回 schema，至少在 server log 里打 debug。

---

## 不建议现在一起做的改动

以下先不要混进这一轮：

1. 改 nginx 配置
2. 改 verifier limiter 逻辑
3. 切 client RR 成默认正式评测
4. 启用 `isolation: lite`
5. 大规模重构 `run_command_bare()`

原因：

- 这会让定位变脏
- 当前最想验证的是：**修掉输出采集竞态后，单 backend 高并发是否明显稳定**

---

## 修复后的验证顺序

按下面顺序验证，避免变量太多：

### Step A：单 backend fixed-response 重判

目标：

- 先只验证 Sandbox 自身是否明显变稳

建议：

- backend: 单个 backend，例如 `8281`
- 数据：`delta69` fixed raw responses
- 并发：`16 / 32 / 64 / 96 / 128 / 180`

成功标准：

- `accepted_mismatch_count` 明显下降
- 最好逼近 `0~1`
- `pass_ratio_mismatch_count` 明显下降

### Step B：单-upstream nginx

目标：

- 确认 nginx 单 upstream 是否与 direct backend 等价

建议：

- `8094 -> 8281`

成功标准：

- 与 direct backend 接近

### Step C：8-backend client RR

目标：

- 再看多 backend 是否还会明显放大问题

成功标准：

- 即使不完全为 0，也应该明显优于当前旧结果

---

## 预期结果

如果这次 patch 命中根因，最可能看到：

1. `same response -> different verdict` 大幅减少
2. `actual=''` / truncated stdout 明显减少
3. 单 backend 与单-upstream 基本稳定
4. 多 backend RR 仍可能略差，但不会再像现在这样明显漂

如果 patch 后问题依旧基本不变，再考虑下一层：

- `isolation: none` 的宿主级干扰
- 多 backend 池的资源竞争
- nginx upstream 路径的额外放大

---

## 推荐执行顺序

1. patch [execution.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/utils/execution.py)
2. patch [base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py)
3. 重启 sandbox pool
4. 先跑单 backend fixed-response 高并发复现
5. 再跑 RR / nginx 对照
6. 稳定后再决定是否切正式评测到 client RR

额外建议在验证时顺手记录：

- backend 进程数是否持续增长
- 是否出现明显 orphan subprocess

这样可以同时确认：

- 输出采集问题是否被修掉
- 正常路径的 descendant cleanup 是否足够

---

## 一句话结论

这轮最小修复的核心不是“让 sandbox 更快”，而是：

**确保子进程结束后，stdout/stderr 被完整、稳定地读出来，再做清理。**

只要这一点没有保证，后面的 verifier、RR、nginx、repair 结论都会继续被污染。
