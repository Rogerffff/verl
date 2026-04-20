**结论**

这次已经基本把问题定位到了：

- **不是模型输出本身**
- **不是单个 sandbox backend 自己不稳定**
- **也不只是“多个 sandbox 并发导致宿主机变慢”**
- **最像的问题是：`nginx` 的“多-upstream LB 路径”在高并发下会偶发返回坏结果**，表现成：
  - 空 `stdout`
  - 少最后几行输出
  - 从而把同一 response 判成 `wrong_answer`

我现在认为，旧的 `raw` vs `repair_raw first-pass` 漂移，**主因就在这里**。

**最关键的证据**

我在 `vastai2` 上重启并拉起了三套池：

- 旧池：`8081..8088 -> 8090`
- 新池 A：`8181..8188 -> 8091`
- 新池 B：`8281..8288 -> 8092`

然后用你之前确实发生过“same response, different verdict”的代表题，做了 4 组对照：

1. **直连单 backend，高并发**
2. **单-upstream LB，高并发**
3. **客户端自己 round-robin 多 backend，高并发**
4. **nginx 多-upstream LB，高并发**

结果非常一致：

**1. 直连单 backend 是稳定的**
对这些最可疑的 case，`100` 次并发请求全稳定：

- `step1000:Codeforces/1064/B:test18`
  - `direct_8181`: `100/100 success`
  - `direct_8182`: `100/100 success`
  - `direct_8184`: `100/100 success`
  - `direct_8287`: `100/100 success`

- `step900:Codeforces/1175/G:test42`
  - `direct_8181`: `100/100 success`
  - `direct_8182`: `100/100 success`
  - `direct_8184`: `100/100 success`
  - `direct_8287`: `100/100 success`

**2. 单-upstream LB 也是稳定的**
我额外起了：
- `8093 -> 8181`
- `8094 -> 8281`

同样 `100` 次高并发：

- `step1000:Codeforces/1064/B:test18`
  - `lb_8093_single`: `100/100 success`
  - `lb_8094_single`: `100/100 success`
- `step900:Codeforces/1175/G:test42`
  - `lb_8093_single`: `100/100 success`
  - `lb_8094_single`: `100/100 success`

**3. 客户端自己 round-robin 多 backend 也稳定**
也就是说，不经过 nginx，直接把请求分发到 `8181..8188` / `8281..8288`：

- `step1000:Codeforces/1064/B:test18`
  - `rr_818x`: `100/100 success`
  - `rr_828x`: `100/100 success`
- `step900:Codeforces/1175/G:test42`
  - `rr_818x`: `100/100 success`
  - `rr_828x`: `100/100 success`

这条很关键，因为它说明：
- **多 backend 同时跑本身不是问题**
- **同一台宿主机上 8 个 sandbox 一起承压也不是充分条件**
- 问题更像是 **经过 nginx 多-upstream 代理这条路径** 才会出现

**4. 只有 nginx 多-upstream LB 在高并发下会复现**
代表性结果：

- `step1000:Codeforces/1064/B:test18`
  - `lb_8091`: `99 success / 1 wrong_answer`
    - 错那次输出从应有的 `19` 字符，变成了 `16` 字符，少了最后一段
  - `lb_8092`: 在一轮实验里 `100/100 success`，另一轮里也出现过 `1` 次异常

- `step900:Codeforces/1175/G:test42`
  - `lb_8091`: `99 success / 1 wrong_answer`
    - 错那次是 **空输出**
  - `lb_8092`: `93 success / 7 wrong_answer`
    - 这 `7` 次全是 **空输出**
    - 而且分布在多个 backend，不是某一个特定 backend 脏掉

这说明：
- 问题不是某个单独 backend 坏了
- 而是 **多-upstream LB 在高并发下偶发把响应搞坏**

**我检查过的代码链路**

重点看了这些文件：

- [verifier/shared.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py)
- [nginx_sandbox_lb.conf.template](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/nginx_sandbox_lb.conf.template)
- [render_nginx_sandbox_lb.sh](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/render_nginx_sandbox_lb.sh)
- [SandboxFusion client.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/scripts/client/src/sandbox_fusion/client.py)
- [SandboxFusion sandbox_api.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/server/sandbox_api.py)
- [SandboxFusion base.py](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/runners/base.py)
- [SandboxFusion local.yaml](/Users/roger/Desktop/coding_RL_project/verl/SandboxFusion/sandbox/configs/local.yaml)

当前最可疑的层是：
- [nginx_sandbox_lb.conf.template](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/phase_2_%20GRPO/ops/nginx_sandbox_lb.conf.template)

它的特点是：
- 多 upstream
- 无 sticky
- `worker_processes auto`
- `proxy_http_version 1.1`
- `Connection ""`

我现在**不敢直接拍板是哪一行配置**，但已经可以很有把握地说：

> **问题出在“多-upstream nginx LB + 高并发”这条路径，不在 verifier 比较逻辑，也不在单 backend SandboxFusion 执行本身。**

**这对当前项目的影响**

1. **旧的 `repair_raw first-pass` 对比不该再信**
   这一点现在基本坐实了。

2. **现在的 `reuse-firstpass` repair 结论仍然可用**
   因为它修掉了 first-pass 漂移这个最大污染源。

3. **但 second-pass judge 仍然可能残留少量 LB 噪声**
   所以：
   - `+1`、`+2` 这种很小的 solve 差异，还是要更谨慎
   - `+12`、`+13` 这种量级，我仍然认为是有意义的

**我建议的下一步**

**优先级最高**
1. **后续所有正式 repair / eval，不再用多-upstream nginx LB 做最终口径**
   两个更稳的选择：
   - 用 **客户端 round-robin 直打 backend**
   - 或用 **单-upstream / 单 backend dedicated eval pool**

**很值得做**
2. 在 verifier 结果里落盘 backend 身份
   现在 nginx 已经有 `X-Upstream-Addr`，可以把它记进 `per_case_results`，以后排查会快很多。

**可以作为修复实验**
3. 单独测试更保守的 nginx 配置
   我还没把这一步完全跑完，但方向是：
   - `worker_processes 1`
   - `proxy_http_version 1.0`
   - `Connection: close`
   - 或 sticky upstream  
   这可以帮助确认是不是 nginx 连接复用/多 worker 路径的问题。

**对你当前主线最直接的建议**
4. **不要因为这个问题暂停后续 SFT 设计**
   更合理的是：
   - repair / ablation / mining 继续做
   - 但正式评测协议改成：
     - `reuse-firstpass`
     - **不用多-upstream nginx LB**

**临时实验产物**
这次深挖的临时结果在：

- 本地 summary：
  - [/tmp/nondet_repro_summary_v2.json](/tmp/nondet_repro_summary_v2.json)
- 本地明细：
  - [/tmp/nondet_repro_results_v2.jsonl](/tmp/nondet_repro_results_v2.jsonl)
- 远端 summary 对应目录：
  - `/tmp/nondet_repro_out_v2/`
- 远端 targeted 结果：
  - `/tmp/nondet_repro_targeted.jsonl`

**一句话拍板**

- **已经基本排清：问题主要在 nginx 多-upstream LB 高并发路径。**
- **不是模型，不是单 backend sandbox，不是 verifier 判分逻辑。**
- 现在最该做的不是暂停项目，而是：
  - **把正式评测从多-upstream nginx LB 切走**
  - 然后继续你的 repair ablation 和后续 SFT 数据收集。

**结论**

现在**不能**直接把正式评测默认切到“客户端 round-robin 直打 backend”。

我已经把 RR 支持补进当前链路，并在远端做了一个更干净的验证：  
用 `delta69` 的**固定 raw response** 重新判题，只测 judge 路径，不让 vLLM 生成漂移混进来。这个测试比再跑一条 full raw eval 更能回答你现在的问题。

代码改动在：
- [shared.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/shared.py)
- [__init__.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/verifier/__init__.py)
- [phase0_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase0_eval.py)
- [phase4_repair_eval.py](/Users/roger/Desktop/coding_RL_project/verl/coding_model_project/src/phase4_repair_eval.py)

现在 `shared verifier` 已支持：
- `--sandbox_url http://a,http://b,http://c`
- 每次真实 `run_code()` RPC 在线程安全 RR 下挑一个 backend
- 老的单 URL 路径自动退化成“取第一个 endpoint”

**验证结果**

我在 `vastai2` 上用同一批 `delta69 raw per_problem` 固定 response，分别重判了 3 条路径，全部都是同一套并发口径：
- `max_concurrent_judges = 180`
- `verifier_limiter_budget = 180`

结果是：

1. `8-backend client RR`，直打 `8181..8188`
- `step900`: `accepted mismatch = 7/69`, `pass_ratio mismatch = 12/69`
- `step1000`: `accepted mismatch = 10/69`, `pass_ratio mismatch = 12/69`

2. `单 backend 直连`，只打 `8181`
- `step900`: `3/69`, `6/69`
- `step1000`: `5/69`, `6/69`

3. `单-upstream nginx`，`8093 -> 8181`
- `step900`: `5/69`, `8/69`
- `step1000`: `5/69`, `6/69`

这说明两件事：

- **RR 直打 backend 在“相同并发级别”下是能跑通的**，不会直接挂掉。
- 但**它在当前 full eval 并发下并不稳定 enough**，因为固定 response 仍然会被判出不同结果。

更关键的是：

- 之前我们已经证明：`nginx 多-upstream LB` 确实会放大问题。
- 但这次新的结果说明：**问题不只在 nginx。**
- 因为即使绕过 nginx，`单 backend` 和 `client RR` 在 `180` 这档 full eval 并发下仍然有 mismatch。

所以现在更准确的判断是：

> `nginx 多-upstream` 是坏上加坏，  
> 但底层 `SandboxFusion + 当前无隔离池 + 高并发混合负载` 本身也还不够稳定。

**补充**

我也尝试起了一条真正的 full raw eval 去测 RR 直连，但那条先被远端 vLLM 启动环境打断了，不是 sandbox judge 自身失败。  
不过这不影响上面的结论，因为这次 fixed-response 重判已经更直接地测到了你真正关心的 judge 稳定性。

**建议**

1. **先不要把正式评测默认切到 client RR。**
2. 后续正式结论继续优先用：
   - `reuse-firstpass`
   - 并且把小幅差异视为不稳定区间
3. 下一步最值得做的是一轮**降并发稳定性扫描**，而不是继续换路由方式：
   - 例如固定 `step900 delta69 raw per_problem`
   - 直接扫：
     - `verifier_limiter_budget = 32`
     - `64`
     - `96`
     - `180`
   - 同时比较：
     - 单 backend
     - client RR
4. 如果目标是“先得到最稳的正式评测方案”，我现在更建议：
   - **单 backend / 单-upstream dedicated eval pool**
   - 配合**更保守的 judge 并发**
   - 而不是立刻全量切 RR
5. 还值得补一个工程改动：
   - 把 backend identity 写进 `per_case_results`
   - 以后这类漂移会容易定位很多

一句话说：

- **client RR 不是坏方案，但在你现在这套 full eval 并发下还不够稳定，不能直接作为默认正式评测方案。**
- 当前更该做的是：**先找出稳定并发区间**，再决定用单 backend 还是 RR。

现在已经推进起来了。前 3 条结果先记一下：

rr_828x @ 16
step900: accepted mismatch 4, pass_ratio mismatch 4, 12.0s
step1000: 5 / 5, 但这条 unusually slow，121.0s
rr_828x @ 32
step900: 4 / 6, 11.0s
我继续让它跑完，等完整表出来再做切换判断。

已运行 sleep 90 && ssh -o ClearAllForwardings=yes vastai2 'echo === ps; ps -o pid,etime,pcpu,pmem,cmd -p 142485 || true; echo === tail; tail -n 200 /workspace/eval_logs/rr_stability_scan.log'（1m 32s）


趋势已经开始出来了，rr_828x 的 mismatch 随 budget 上升明显变坏，96 已经跳到 9 / 12。我继续等整轮扫完，再和单 backend / 单-upstream做完整对比。

已运行 sleep 120 && ssh -o ClearAllForwardings=yes vastai2 'echo === ps; ps -o pid,etime,pcpu,pmem,cmd -p 142485 || true; echo === tail; tail -n 260 /workspace/eval_logs/rr_stability_scan.log'（2m 2s）


rr_828x 这一整列已经跑完了，结论很强：

step900: mismatch 从 16 档的 4/4 上升到 180 档的 10/14
step1000: 5/5 到 8/10
也就是说，RR 在高并发下确实会持续变差。我继续等后面的 direct_8281 和 lb_8094_single，这样就能知道它相对单 backend 的收益/代价到底值不值。

已运行 sleep 120 && ssh -o ClearAllForwardings=yes vastai2 'echo === ps; ps -o pid,etime,pcpu,pmem,cmd -p 142485 || true; echo === tail; tail -n 320 /workspace/eval_logs/rr_stability_scan.log'（2m 2s）


第二列也开始出了。direct_8281 @ budget=16 已经出来了：

step900: accepted mismatch 4, pass_ratio mismatch 7, 但耗时 46.6s
这说明：

单 backend 确实更慢很多
但它也不天然更稳，至少在这第一格上没有明显赢 RR
我继续等完整 direct_8281 列。

已运行 sleep 150 && ssh -o ClearAllForwardings=yes vastai2 'echo === ps; ps -o pid,etime,pcpu,pmem,cmd -p 142485 || true; echo === tail; tail -n 360 /workspace/eval_logs/rr_stability_scan.log'（2m 32s）


direct_8281 到 budget=32 也出来了，当前信号还是一致：

单 backend 慢很多
稳定性没有明显优于 RR
step900: 4/7
step1000: 9/10
这基本已经说明“只靠切到单 backend”未必能解决问题，但我还是继续等完整扫描跑完，避免过早下结论。
