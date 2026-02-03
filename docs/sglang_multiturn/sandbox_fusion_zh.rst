===============================
Sandbox Fusion 工具集成
===============================

最后更新日期: 2025年6月10日

动机 (Motivations)
=================

- 作为 verl 的用户，我们希望允许模型在 Actor 展开 (rollout) 期间调用某些工具，并将结果纳入训练过程。
- 字节跳动的一位同事提出了一篇论文，旨在通过代码执行工具增强模型能力。
- 我们的目标是支持推理引擎的工具调用能力，使用 `sandbox-fusion` 作为代码执行系统，为社区提供 `retools` 的重新实现。

通过 Sandbox Fusion + FaaS 集成计算奖励
=======================================

- 在当前的数据集和任务中，已经存在类似的工作（例如 Prime），它使用本地进程作为运行器 (runner) 来执行模型生成的代码以进行奖励计算。
- 在此基础上，#1429 通过集成 FaaS 作为奖励计算的运行器，进一步推进了设计。

目标 (Goals)
============

- 适配 `sglang` 工具调用协议并为 sandbox fusion 定义工具。
- 与 `async-rollout` 过程集成，确保 sandbox fusion 工具遵循 asyncIO 惯例。
- 设计并实现一个基础的限流器 (rate limiter)，以防止诸如 429 错误之类的问题。

非目标 (Non-Goals)
==================

- 训练效果不在讨论范围内。
- 不考虑可观测性指标。
- 不涉及分布式故障转移和组件容错。

设计细节 (Design Details)
========================

工具 Schema 定义
----------------

- 目前仅考虑代码执行，要求模型提供的 JSON 中包含 `code` 字段。
- 目前仅支持 Python 代码，因此未定义 `language` 参数。

.. code-block:: python

   OpenAIFunctionToolSchema(
       type="function",
       function=OpenAIFunctionSchema(
           name="code_interpreter",
           description="一个用于执行代码的工具。",
           parameters=OpenAIFunctionParametersSchema(
               type="object",
               properties={
                   "code": OpenAIFunctionPropertySchema(
                       type="string",
                       description="要执行的代码。",
                       enum=None,
                   )
               },
               required=["code"],
           ),
           strict=False,
       )
   )

配置参数
--------

+----------------------------+--------------------------------------------------------------+
| 参数名称                    | 描述                                                         |
+============================+==============================================================+
| `num_workers`              | 每个 DP 请求运行器时使用的辅助线程/进程数。                  |
+----------------------------+--------------------------------------------------------------+
| `rate_limit`               | 全局并发代码执行限制。默认值：10                             |
+----------------------------+--------------------------------------------------------------+
| `default_timeout`          | 每次代码执行的超时时间（秒）。默认值：30                     |
+----------------------------+--------------------------------------------------------------+
| `default_language`         | 默认编程语言。默认值："python"                               |
+----------------------------+--------------------------------------------------------------+
| `enable_global_rate_limit` | 是否启用全局限流。默认值：True                               |
+----------------------------+--------------------------------------------------------------+
| `sandbox_fusion_url`       | veFaas 沙箱执行服务的 URL                                    |
+----------------------------+--------------------------------------------------------------+

限流设计
--------

目标：

- 使用令牌桶模型限制正在处理的请求数量。
- 确保按顺序提交给代码运行器，以避免因退避 (backoff) 导致的饥饿问题。

设计亮点：

- 使用 Ray Global Actor 作为集群级别的单例分布式计数器。
- 使用信号量 (Semaphore) 进行计数，在独立的线程池中进行 `acquire` 和 `release` 以保持顺序。
- 使用 Ray 的 cloud-pickle 序列化函数，以实现 `ExecutionWorker` 的解耦。

.. code-block:: python

   @ray.remote(concurrency_groups={"acquire": 1,"release": 10})
   class TokenBucketWorker:
       def __init__(self, rate_limit: int):
           self.rate_limit = rate_limit
           self.current_count = 0
           self._semaphore = threading.Semaphore(rate_limit)

       @ray.method(concurrency_group="acquire")
       def acquire(self):
           self._semaphore.acquire()
           self.current_count += 1

       @ray.method(concurrency_group="release")
       def release(self):
           self._semaphore.release()
           self.current_count -= 1

       def get_current_count(self):
           return self.current_count

   class ExecutionWorker:
       def __init__(self, enable_global_rate_limit=True, rate_limit=10):
           self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

       def _init_rate_limit(self, rate_limit):
           return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

       def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
           with ExitStack() as stack:
               stack.callback(self.rate_limit_worker.release.remote)
               ray.get(self.rate_limit_worker.acquire.remote())
               try:
                   return fn(*fn_args, **fn_kwargs)
               except Exception as e:
                   logger.warning(f"执行代码时出错: {e}")

   def init_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode=PoolMode.ThreadMode):
       if mode == PoolMode.ThreadMode:
           return ray.remote(ExecutionWorker).options(max_concurrency=num_workers).remote(
               enable_global_rate_limit=enable_global_rate_limit,
               rate_limit=rate_limit
           )
       else:
           raise NotImplementedError("进程模式尚未实现")

工具实现
--------

- 使用 `instance_id` 在多轮对话中标识请求。
- 使用 `execution_pool` 实现异步调用。
- 在 rollout 完成后清理状态。

.. code-block:: python

   class SandboxFusionTool(BaseTool):
       def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
           ...
           self.execution_pool = init_execution_pool(...)
           ...

       async def create(self, instance_id: Optional[str] = None, ...):
           ...

        async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
            code = parameters.get("code", "")
            timeout = parameters.get("timeout", self.default_timeout)
            language = parameters.get("language", self.default_language)
            if not isinstance(code, str):
                code = str(code)

            result = await self.execution_pool.execute.remote(self.execute_code,instance_id,code,timeout,language)
            self._instance_dict[instance_id]["reward"].append(result.strip())

            return result, result, {}

        def execute_code(self,instance_id,code,timeout=30,language="python"):
            result_status, metadata  = _process_single_case(0, None, None,self.sandbox_fusion_url, code, timeout, language)
            # 我们应该始终期待这个结果，因为我们没有正确答案
            if metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                return actual_output
            else:
                return "no stdout here"

       async def calc_reward(self, instance_id: str, ...):
           ...

       async def release(self, instance_id: str, ...):
           ...

测试计划 (Test Plan)
===================

单元测试
--------

- **test_tools_registration**: 测试工具注册和初始化。
- **test_rollout_req_creation**: 验证 `AsyncRolloutReq` 是否正确构建。
- **test_over_size_case**: 确保在超过 `max_seq_len` 时 rollout 提前终止。
- **test_tool_call_basic_case**: 模拟 `sglang` 输出，验证工具调用和结果。
- **test_tool_call_batch_case**: 测试工具调用的批量处理。
- **test_basic_multi_process_init**: 验证 Ray global actor 的单例行为。
- **TestSingleNodeRateLimiterCase**: 验证限流器在单节点模式下工作。
- **test_rotten_execution**: 确保限流器能从函数错误中恢复。
- **TestMultiNodeRateLimiterCase**: 验证在多节点环境中的行为。

端到端 (e2e) 测试
-----------------
我们在 `tests/special_e2e` 文件夹中提供了 e2e 测试脚本，名为 `tests/special_e2e/run_gsm8k_fsdp_sgl_multiturn_sf_tool.sh`

通过设置 'trainer.rollout_data_dir'，您可以将 rollout 数据转储到本地磁盘。以下是一个从 rollout 数据中提取的示例：

.. code-block:: python

   {
     "input": "
     
     system\n你是一名数学专家。你会得到一个问题，你需要一步步解决它。在进行任何工具调用之前，请先进行一步步的推理。你应该在一步步解决问题后、生成最终答案之前，至少调用一次 `calc_gsm8k_reward` 工具，并在必要时完善你的答案。将你的最终答案放在 `#### <answer>` 格式中。\n\n# 工具\n\n你可以调用一个或多个函数来辅助用户查询。\n\n在 <tools></tools> XML 标签内为你提供了函数签名：\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"code_interpreter\", \"description\": \"一个用于执行代码的工具。\", \"parameters\": {\"type\": \"object\", \"properties\": {\"code\": {\"type\": \"string\", \"description\": \"要执行的代码。\", \"enum\": null}}, \"required\": [\"code\"]}, \"strict\": false}}\n</tools>\n\n对于每个函数调用，在 <tool_call></tool_call> XML 标签内返回一个带有函数名和参数的 json 对象：\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n
     
     user\n约翰每年都会得到同样比例的奖金。去年他赚了 100,000 美元，得到了 10,000 美元的奖金。今年他赚了 200,000 美元。约翰今年的总工资加上奖金一共是多少？让我们一步步思考，并在 `####` 之后输出最终答案。\n
     
     assistant\n",
     "output": "assistant\n<think>\n好的，让我看看。约翰每年得到的奖金比例是相同的。去年，他赚了 100,000 美元，得到了 10,000 美元的奖金。今年，他赚了 200,000 美元。问题是问他今年的总工资加上奖金一共是多少。\n\n首先，我需要计算出他得到的奖金比例。既然奖金比例每年都一样，如果我能根据去年的细节找到那个比例，我就可以把它应用到今年的收入上，算出今年的奖金。然后，我会把那笔奖金加到他今年的总工资中。\n\n去年，他赚了 100,000 美元，得到了 10,000 美元的奖金。为了找到比例，我可以用奖金除以他的总收入。所以，10,000 除以 100,000。让我计算一下。\n\n10,000 / 100,000 等于 0.1，也就是 10%。所以，他每年的奖金是工资的 10%。\n\n今年，他赚了 200,000 美元。所以，今年的奖金将是 200,000 的 10%。让我使用代码解释器来计算，以确保准确性。\n
     
     <tool_call>\n{\"name\": \"code_interpreter\", \"arguments\": {\"code\": \"total_pay_this_year = 200000\\nbonus_percentage = 10 / 100\\nbonus_this_year = total_pay_this_year * bonus_percentage\\ntotal_income_this_year = total_pay_this_year + bonus_this_year\\nprint(total_income_this_year)\", \"executes\": \"True\"}}\n</tool_call>\n
     
     tool\n220000.0\n\n
     
     assistant\n<think>\n代码显示约翰今年的总收入将是 220,000 美元。这与去年 100,000 美元工资对应 10% 奖金（即 10,000 美元）的情况是一致的。因此，今年的奖金 20,000 美元加上他的总工资 200,000 美元，总收入为 220,000 美元。\n\n我需要验证是否还有其他因素。题目说明每年是恒定的百分比奖金，所以没有提到改变百分比或其他变量。因此，直接对今年的部分工资数额应用 10% 的比例将得到 20,000 美元。将其加到他的总工资中得到 220,000 美元。计算正确且符合问题的参数。\n</think>\n\n#### 220000.0",
     "score": 0,
     "step": 1
   }

以下是可读格式版本：

.. code-block:: python

   [system]
   
   你是一名数学专家。你会得到一个问题，你需要一步步解决它。在进行任何工具调用之前，请先进行一步步的推理。你应该在一步步解决问题后、生成最终答案之前，至少调用一次 `calc_gsm8k_reward` 工具，并在必要时完善你的答案。将你的最终答案放在 `#### <answer>` 格式中。
   
   # 工具
   
   你可以调用一个或多个函数来辅助用户查询。
   
   在 <tools></tools> XML 标签内为你提供了函数签名：
   <tools>
   {"type": "function", "function": {"name": "code_interpreter", "description": "一个用于执行代码的工具。", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "要执行的代码。", "enum": null}}, "required": ["code"]}, "strict": false}}
   </tools>
   
   对于每个函数调用，在 <tool_call></tool_call> XML 标签内返回一个带有函数名和参数的 json 对象：
   <tool_call>
   {"name": <function-name>, "arguments": <args-json-object>}
   </tool_call>
   
   [user]
   
   约翰每年都会得到同样比例的奖金。去年他赚了 100,000 美元，得到了 10,000 美元的奖金。今年他赚了 200,000 美元。约翰今年的总工资加上奖金一共是多少？让我们一步步思考，并在 `####` 之后输出最终答案。
   
   [assistant]
   
   <think>
   好的，让我看看。约翰每年得到的奖金比例是相同的。去年，他赚了 100,000 美元，得到了 10,000 美元的奖金。今年，他赚了 200,000 美元。问题是问他今年的总工资加上奖金一共是多少。
   
   首先，我需要计算出他得到的奖金比例。既然奖金比例每年都一样，如果我能根据去年的细节找到那个比例，我就可以把它应用到今年的收入上，算出今年的奖金。然后，我会把那笔奖金加到他今年的总工资中。
   
   去年，他赚了 100,000 美元，得到了 10,000 美元的奖金。为了找到比例，我可以用奖金除以他的总收入。所以，10,000 除以 100,000。让我计算一下。
   
   10,000 / 100,000 等于 0.1，也就是 10%。所以，他每年的奖金是工资的 10%。
   
   今年，他赚了 200,000 美元。所以，今年的奖金将是 200,000 的 10%。让我使用代码解释器来计算，以确保准确性。
   <tool_call>
   {"name": "code_interpreter", "arguments": {"code": "total_pay_this_year = 200000\nbonus_percentage = 10 / 100\nbonus_this_year = total_pay_this_year * bonus_percentage\ntotal_income_this_year = total_pay_this_year + bonus_this_year\nprint(total_income_this_year)", "executes": "True"}}
   </tool_call>
   
   [tool]
   
   220000.0
   
   [assistant]
   
   <think>
   代码显示约翰今年的总收入将是 220,000 美元。这与去年 100,000 美元工资对应 10% 奖金（即 10,000 美元）的情况是一致的。因此，今年的奖金 20,000 美元加上他的总工资 200,000 美元，总收入为 220,000 美元。
   
   我需要验证是否还有其他因素。题目说明每年是恒定的百分比奖金，所以没有提到改变百分比或其他变量。因此，直接对今年的部分工资数额应用 10% 的比例将得到 20,000 美元。将其加到他的总工资中得到 220,000 美元。计算正确且符合问题的参数。
   </think>
   
   #### 220000.0


您也可以使用 `RolloutViewer` TUI 工具来查看转储的 rollout 数据：


.. code-block:: bash

    python scripts/rollout_viewer.py ${trainer.rollout_data_dir}


.. image:: https://github.com/user-attachments/assets/e34e5157-2880-4a21-afb2-73885d0dfb11
   :alt: RolloutViewer 截图

