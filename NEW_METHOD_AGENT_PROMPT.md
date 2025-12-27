# 给新 Agent 的任务 Prompt：在本仓库中等价接入一个新 Method（通用模板）

> 日期：2025-12-27
>
> 你的任务是把“一个新的方法（method）”从外部项目/论文代码中**等价移植**到本仓库，并让它能被本仓库的 benchmark 主程序调用。
>
> **重要：**你必须把新 method 当作一个黑盒研究原实现，严格保持其 workflow / 数据流 / prompt / 推理逻辑一致，只做“最薄的一层适配”。任何为了“更简洁/更快/更少文件”的改写都可能改变行为，属于不合规（也有学术一致性风险）。

---

## 0. 你要接入的对象（用占位符）

- 新方法名称：`<METHOD_NAME>`（例如：`foo_method`）
- 数据集名称：`<DATASET_NAME>`（例如：`medqa`；当前仓库用 `--dataset_name` 指定）
- 该方法的基座模型：`<BASE_MODEL_NAME>`（例如：`gpt-4o-mini` 或其它；用于结果命名/记录）
- 该方法来自的外部源码目录：`<UPSTREAM_SOURCE_DIR>`

你可以把 `<METHOD_NAME>` / `<DATASET_NAME>` 视为未来会不断替换的变量，本 Prompt 不应绑定任何特定 method 或特定数据集实现。

补充说明（非常重要）：

- 本仓库当前已经接入的 `autogen`、`dylan`、以及 `medqa` 数据集都只是**范例/样例**，用来演示“如何以最薄适配接入一个 method”。
- 你接下来要接入的会是另一个新的 `<METHOD_NAME>`，请按同样的模式推进；不要把本 Prompt 理解为“只为 autogen/medqa 编写”。

---

## 1. 成功标准（验收口径）

完成后必须同时满足：

1. **主程序可调用**：`python -m main --model <METHOD_NAME> --dataset_name <DATASET_NAME> --num_samples 3` 可以运行到结束并输出 Accuracy。
2. **输出规范**：结果写到 `output/` 下，文件名包含 `method + base_model + dataset + timestamp`（主程序已按该格式生成；你需要保证 `<BASE_MODEL_NAME>` 可被合理取得/填充）。
3. **样本数一致**：输出 JSON 内的条目数 == 实际处理的样本数（主程序当前是“每个样本都写一条记录”）。
4. **不污染主程序**：任何 method 特有的配置（API key/base_url、专用 config、prompt、特殊依赖）都必须收敛在 method 自己的目录/封装中，不能把 method-specific 逻辑塞进 `main.py`。
5. **等价移植**：你不能“重写算法”，只能做路径/import/入口函数/配置读取的薄适配。
6. **终端输出克制**：不要在推理循环里打印大量 debug。主程序目前会对部分 method 调用做 stdout/stderr 重定向；你在 wrapper 里也不要额外 print。

---

## 2. 绝对硬约束（必须遵守）

- **禁止改动上游逻辑以求简化**：
  - 不要“把多个模块合并成一个文件”。
  - 不要“删掉看起来没用的中间步骤”。
  - 不要“重写 prompt 文本 / role / system 指令 / tool schema / agent 对话策略”。
  - 不要“为了兼容本仓库而改掉上游默认超参/搜索策略/投票策略/多 agent 协作流程”。

- **路径可移植性**：
  - 不允许在新代码里写死绝对路径（如 `C:\...`）。
  - 所有文件路径都必须能基于本仓库根目录（`root_path`）拼出来。

- **main.py 必须 method-agnostic**：
  - `main.py` 只能做“选择 method → 调用 method 的统一入口 → 统计/写文件”。
  - 不允许在 `main.py` 里加入“某个 method 才需要的 API 配置读取/注入逻辑”。

- **API 配置只能在 method 内部处理**：
  - 若 method 需要 API key/base_url，必须在 method 内部实现“当 env 缺失时，从 `model_api_configs/model_api_config.json` 回退读取并设置 env”的逻辑。
  - 主程序不负责注入这些信息。

---

## 3. 本仓库约定（你需要遵循的落盘结构）

### 3.1 method 目录结构

把新 method 放到：

- `methods/<METHOD_NAME>/...`（你可以尽量保持上游项目的相对结构，不要乱重排）

并提供一个最薄的、明确的入口文件：

- `methods/<METHOD_NAME>/infer_<DATASET_NAME>.py`

入口函数命名建议：

- `<METHOD_NAME>_infer_<DATASET_NAME>(question: str, root_path: str) -> str`

> 说明：主程序只需要一个“给定 question 返回模型最终选择/答案”的接口；method 内部怎么做多轮、多 agent、工具调用等都保持上游一致。

### 3.2 dataset 抽象层

本仓库已有 `dataset_utils/`：

- `load_test_split(dataset_path_name, dataset_name)`
- `format_question(sample, dataset_name)`
- `extract_choice(text, dataset_name)`

你的新 method **不要**自行在主程序层面解析数据集；它应该只接收 `question` 字符串。若 method 强依赖“选项结构/答案字母”等，请把必要的信息编码在 `format_question` 的输出里，或只在 method wrapper 内做**最小补齐**，但不要改变上游核心行为。

---

## 4. 实施步骤（严格按顺序做）

### Step 1：导入上游代码（保持原样）

- 将 `<UPSTREAM_SOURCE_DIR>` 中与 `<METHOD_NAME>` 相关的源码拷贝到 `methods/<METHOD_NAME>/`。
- 尽量保持文件相对路径、配置文件名、prompt 文本不变。

### Step 2：修正 import 与相对路径（只做必要改动）

- 将上游代码里的包导入路径调整为在本仓库内可 import。
- 任何读取本地文件的地方（prompt/config/模板等），都改为：
  - `Path(root_path) / "methods" / "<METHOD_NAME>" / ...`
  - 或基于 `Path(__file__).resolve().parent`（更推荐）

> 目标：代码在任意机器/任意工作目录下运行都不依赖绝对路径。

### Step 3：实现 method 的统一入口（infer）

在 `methods/<METHOD_NAME>/infer_<DATASET_NAME>.py` 中新增：

- `<METHOD_NAME>_infer_<DATASET_NAME>(question: str, root_path: str) -> str`

要求：

- 输入就是 `question` 字符串（来自 `dataset_utils.format_question`）。
- 返回一个最终答案（字母/选项文本/可被 `dataset_utils.extract_choice` 抽取的形式）。
- wrapper 内只负责“把 question 塞到上游的原始运行入口里”，不要改其内部 workflow。

### Step 4：处理 API 配置回退（如果该 method 走 API）

如果 `<METHOD_NAME>` 调用 OpenAI 兼容 API（或其它 HTTP API）：

- 在 method 内实现一个函数：
  - 当 `OPENAI_API_KEY` / `BASE_URL`（或上游所需 env）不存在时，从 `model_api_configs/model_api_config.json` 读取并注入。
- 读取路径必须基于 `root_path`：
  - `Path(root_path) / "model_api_configs" / "model_api_config.json"`

并确保：

- 缺配置时抛出**清晰错误**（告诉用户缺哪个字段、配置文件在哪）。
- 不要在 `main.py` 写任何注入逻辑。

### Step 5：把新 method 接到主程序（最小改动）

由于当前 `main.py` 使用 argparse 的 `--model` choices + `if/elif` 分支调度，你需要做**最小改动**：

- `from methods.<METHOD_NAME> import <METHOD_NAME>_infer_<DATASET_NAME>`（或从其模块导入）
- 在 `--model` 的 `choices` 里加入 `<METHOD_NAME>`
- 在推理分支增加：
  - `final_decision = <METHOD_NAME>_infer_<DATASET_NAME>(question, args.root_path)`

注意：

- 不要把 `<METHOD_NAME>` 的配置解析、prompt 读取等塞进 `main.py`。
- 若该 method 会大量打印，主程序可对其调用做 stdout/stderr 重定向；你不要在 method 内大改打印行为。

### Step 6：基座模型名（用于输出命名）

主程序会把 `base_model_name` 写进输出文件名（以及你可能写入 results 字段）。你需要保证：

- `<BASE_MODEL_NAME>` 能被合理获得：
  - 如果上游有 `configs/config_main.yaml` 或类似配置，建议在 method 内提供一个“读取 base model 名”的小函数供主程序使用，或让主程序按现有约定读取（尽量少改主程序）。
  - 如果无法读取，就用一个稳定的默认值（例如 `unknown`），但要在 README/说明里说清。

> 注意：这里“只影响命名/记录”，不应改变推理行为。

### Step 7：验证（必须做）

请至少跑两条命令：

- `python -m main -h`
  - 必须能正常输出 help（不能因为可选依赖缺失在 import 阶段崩溃）。
- `python -m main --model <METHOD_NAME> --dataset_name <DATASET_NAME> --num_samples 3`
  - 必须跑完并生成一个新的 `output/*.json`。

### Step 8：自检清单（提交前最后一遍）

- [ ] 新增代码没有任何绝对路径
- [ ] `main.py` 没有新增 method-specific 配置注入
- [ ] method 的 wrapper 只做薄适配，没有改 prompt / workflow
- [ ] 输出 JSON 条目数 == num_samples（或实际处理数）
- [ ] 终端没有刷屏（只有 tqdm + 最终 accuracy）

---

## 5. 常见坑（请主动规避）

1. **导入即执行**：
   - 不要在模块 import 时就读取大模型、加载巨大权重、或发起 API 请求。
   - 这会导致 `python -m main -h` 也崩。

2. **可选依赖**：
   - 如果上游依赖某些可选包（例如某些 VL 工具），请改成“用到时再 import”，或给出清晰错误信息。

3. **数据集耦合**：
   - method wrapper 只接收 `question` 字符串。
   - 数据加载、样本遍历、统计、写文件都在主程序。

4. **静默 vs 改逻辑**：
   - 需要减少输出时，优先由主程序重定向 stdout/stderr。
   - 不要删除上游内部日志/print（除非确定不影响行为，且用户明确允许）。

---

## 6. 示范性展示：以 `autogen` → `medqa` 为例（仅用于演示模式）

这一节的目的：用一个**已接入的范例**（`autogen` 方法 + `medqa` 数据集）展示“应该长什么样”。你在接入新 `<METHOD_NAME>` 时，照着这个形态做即可。

### 6.1 范例的落盘形态

你应该能在仓库里看到类似结构（具体文件可能略有不同，以实际为准）：

- `methods/autogen/`：上游方法主体代码（尽量保持原样）
- `methods/autogen/configs/`：该方法自己的配置（例如 `config_main.yaml`）
- `methods/autogen/infer_medqa.py`：对接本仓库 benchmark 的薄入口

其中关键点是：

- 入口函数为 `autogen_infer_medqa(question: str, root_path: str) -> str`
- 它只负责把 `question` 交给 autogen 原始流程，并返回最终答案/选项
- API key / base_url 等配置（如果需要）应在 `methods/autogen/` 内部处理回退读取，而不是在 `main.py`

### 6.2 主程序如何调用（命令示例）

在仓库根目录下运行：

```bash
python -m main --model autogen --dataset_name medqa --num_samples 3
```

预期现象：

- 终端主要显示 tqdm 进度条，结束后打印 Accuracy
- 在 `output/` 下生成一个结果文件，文件名包含：`autogen + base_model + medqa + timestamp`

### 6.3 这一范例反推“新 method 应该怎么做”

把 `autogen` 替换成你的 `<METHOD_NAME>`，把 `infer_medqa.py` 替换成 `infer_<DATASET_NAME>.py`：

1. `methods/<METHOD_NAME>/`：放上游源码（尽量不动）
2. `methods/<METHOD_NAME>/infer_<DATASET_NAME>.py`：新增 `<METHOD_NAME>_infer_<DATASET_NAME>(question, root_path)`
3. `main.py`：只做三处最小接线
  - `--model` choices 增加 `<METHOD_NAME>`
  - import 入口函数
  - 增加一个 `elif args.model == "<METHOD_NAME>": final_decision = ...`

注意：这段“示范”只展示模式，不要求你的新 method 也叫 autogen，也不要求你的新数据集也叫 medqa。

---

## 7. 你交付的最小文件集合（建议）

- `methods/<METHOD_NAME>/`（上游方法源码 + configs + prompts 等）
- `methods/<METHOD_NAME>/infer_<DATASET_NAME>.py`（统一入口函数）
- （如需）`methods/<METHOD_NAME>/__init__.py`（便于 import）
- （如需新增依赖）更新 `requirements.txt`

---

## 7. 给你的一句话提醒

**宁可多保留上游代码、少做“聪明重构”，也不要为了漂亮把行为改掉。**
