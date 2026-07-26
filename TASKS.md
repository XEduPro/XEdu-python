# XEdu-python 升级任务清单

维护说明：本文件是当前升级工作的唯一任务真相源。状态只允许基于实际验证结果标注——
「完成」必须能被 import 检查、pytest 或手动运行验证；不允许仅因为文件被创建就标记完成。

状态取值：`done` / `in_progress` / `todo`

每个任务包含三部分：
- **为什么做**：动机和背景，尽量引用这次审计中发现的具体问题（文件、行号、真实 bug）
- **怎么做**：具体到文件路径、函数名、步骤的操作方式
- **验收标准**：怎样算真正做完，而不是"文件存在"

---

## Phase 0：基础设施（模型生态的地基）

### 任务 1：梳理现有模型清单
**状态**：done

**为什么做**
项目里散落着 27+ 个预训练模型，但没有任何一份文档统一记录"有哪些模型、默认文件名是什么、从哪下载、依赖什么"。这导致两个直接后果：
- 无法一眼看出哪些模型的下载配置是错的（例如后面发现的 `pose_face106` bug，如果早有清单会更快被发现）
- 新增模型时没有参照标准，容易漏填字段

**怎么做**
逐一读取 [XEdu/hub/workflow.py](XEdu/hub/workflow.py) 里的 `task_dict`（任务名 → 文件名映射）和 `model_name_map_download`（任务名 → 下载 URL 映射），交叉核对两者是否一致，同时记录每个任务的输入输出类型、依赖包。整理成 [MODELS_INVENTORY.md](MODELS_INVENTORY.md)。

**验收标准**
- [x] 文档覆盖 `task_dict` 里全部任务
- [x] 标注了已知问题（如 pose_face106 URL 错误）
- [x] 标注了每个任务的可选依赖

---

### 任务 2：建立模型注册表 `model_registry.py`
**状态**：done

**为什么做**
模型元数据（URL、文件名、输入输出类型、依赖）原来直接硬编码在 `workflow.py` 的多个 if/elif 分支里，散落在至少 4 个不同位置（`task_dict`、下载 URL 字典、`_load_model` 分支、`format_output` 分支）。这带来两个问题：
1. 改一个模型的配置要同步改好几处，容易漏改（`pose_face106` bug 的根源正是这种散落结构）
2. 无法被程序化查询——用户想知道"这个任务需要装什么依赖"只能去读源码

**怎么做**
新建 [XEdu/hub/model_registry.py](XEdu/hub/model_registry.py)：
- 定义 `ModelMetadata` dataclass，字段覆盖 task_name/model_id/filename/source_url/input_type/output_type/optional_dependencies/auto_download/tags 等
- 定义 `ModelRegistry` 类做注册和查询（`get_model`/`get_models_by_task`/`get_default_model`/`list_tasks`）
- 用 `register_model()` 把 `workflow.py` 里现有的模型逐条注册进去，**URL 必须从 workflow.py 原文抄录，不能编造**
- 统一模型源策略：ModelScope 作为主源，OpenXLab/OpenInnoLab 或 GitHub 原始地址进入 `mirror_urls`，每个自动下载的 ONNX 文件记录 SHA256 校验和

审计中发现并修复的两处伪数据（说明"抄录"这一步必须做，不能图快编数据）：
- `gen_style_*` 系列注册时 URL 写成了占位符 `.../creator/...&name=...`，后来对照 [workflow.py:349-354](XEdu/hub/workflow.py:349) 换成真实 URL
- `segment_anything` 只注册了一条记录、文件名写成假的 `"seg_sam"`，而它实际需要 encoder+decoder 两个文件，后改成两条独立记录 `segment_anything-encoder`/`segment_anything-decoder`

**验收标准**
- [x] 27 个任务全部注册
- [x] `test_no_placeholder_or_malformed_source_urls` 测试通过（防止占位符 URL 再次混入）
- [x] `pose_face106` 有注册记录，且 `auto_download=False`，理由写在 description 里
- [x] 27 个可自动下载的 ONNX 文件已上传到 ModelScope 仓库 `wht0926/xedu-hub-models`，注册表主源指向 ModelScope，原 OpenXLab/OpenInnoLab/GitHub 地址作为备用源

---

### 任务 3：建立模型下载与缓存管理器 `model_store.py`
**状态**：done（已接入 handler 路径、未迁移的 legacy `Workflow` 默认任务路径，以及 `segment_anything` 双模型下载路径）

**为什么做**
原来的下载逻辑（`Downloader` 类，[workflow.py:143](XEdu/hub/workflow.py:143)）每个任务各写一遍调用代码，缓存目录、校验逻辑不统一，且没有 checksum 校验——下载中断或文件损坏后不会被发现，只会在推理时报出难以理解的 ONNX 错误。

**怎么做**
新建 [XEdu/hub/model_store.py](XEdu/hub/model_store.py)：
- `ModelStore._resolve_cache_dir()`：统一缓存目录解析（`XEDU_HOME` > `XEDU_CACHE_DIR` > 平台默认）
- `get_model_path(model_id, auto_download=True)`：查 registry，本地存在就返回，不存在按 `auto_download` 决定是否下载
- `download()` + `_verify_checksum()`：下载后校验 SHA256，失败则删除残留文件并抛 `XEduModelDownloadError`
- `download_from_sources()`：先尝试 ModelScope 主源，再顺序回退到 `mirror_urls`
- `check_dependencies(model_id)`：检查 `optional_dependencies` 是否已安装，缺失则抛 `XEduDependencyError`

**验收标准**
- [x] 缓存目录解析有单测（`test_cache_dir_resolution`）
- [x] 单例模式验证（`test_get_model_store_singleton`）
- [x] 依赖检查逻辑有单测
- [x] 主源失败后回退镜像源有单测（`test_get_model_path_falls_back_to_mirror`）
- [x] legacy `Workflow` 默认任务也通过注册表/ModelStore 下载，不再使用旧硬编码 URL 表
- [x] 新增代码里的 `print()` 已替换为 `logging.getLogger("XEdu.hub.model_store")`

---

### 任务 4：定义统一异常体系 `exceptions.py`
**状态**：done（异常类型已定义；全项目切换到统一异常仍是未完成的整合工作）

**为什么做**
原来出错时的做法很不一致：有的地方 `raise ValueError`，有的地方裸 `except:` 吞掉异常再 `print` 提示，有的地方直接让 `ImportError` 原样冒出。用户没办法用一个 `except XEduError` 统一捕获所有"这是 XEdu 自己的错误"，只能逐个猜测异常类型。

**怎么做**
新建 [XEdu/hub/exceptions.py](XEdu/hub/exceptions.py)：
- 基类 `XEduError(Exception)`
- 按语义分类的子类：`XEduConfigError`/`XEduModelNotFoundError`/`XEduModelDownloadError`/`XEduDependencyError`/`XEduInputError`/`XEduInferenceError`/`XEduOutputError`/`XEduAPIError`（含 `XEduAuthenticationError`/`XEduRateLimitError`)/`XEduNotSupportedError`
- 两个便利函数 `make_dependency_error()`/`make_model_not_found_error()`，生成带修复建议的错误信息（不只告诉用户"错了"，还告诉"怎么修"）

**验收标准**
- [x] 异常继承结构测试通过（`test_exception_hierarchy`）
- [x] 便利函数测试通过
- [ ] **未完成**：`workflow.py`、`repo_model.py` 等旧代码里的 `ValueError`/`RuntimeError`/裸 `except:` 尚未切换成用这套异常体系抛出。这是一项独立的、跨越很多文件的整合工作，暂不在当前优先级里。

---

### 任务 6：补 pytest 骨架与 GitHub Actions CI
**状态**：done

**为什么做**
项目原来没有任何自动化测试。审计过程中至少两次靠"手工跑一遍才发现"的方式抓到真实 bug（打包漏子包、`handlers/base.py` 相对导入写错），这类问题本该由 CI 在提交时就拦下来，而不是靠人肉复查。

**怎么做**
- 新增 [tests/test_registry.py](tests/test_registry.py)、[tests/test_exceptions.py](tests/test_exceptions.py)、[tests/test_model_store.py](tests/test_model_store.py)、[tests/test_handlers.py](tests/test_handlers.py)、[tests/test_workflow_smoke.py](tests/test_workflow_smoke.py)、[tests/test_packaging.py](tests/test_packaging.py)
- 新增 [.github/workflows/ci.yml](.github/workflows/ci.yml)：Python 3.8/3.9/3.10/3.11/3.12 矩阵，跑 `pip install -e ".[dev]"` + `pytest` + wheel 构建
- 修复 `pyproject.toml` 里 `addopts = "-v --cov=..."` 导致本地裸 `pytest`（未装 `pytest-cov`）直接报错退出的问题——覆盖率参数移到 CI 步骤里单独跑，不作为默认行为
- `test_packaging.py::test_wheel_contains_all_subpackages` 标记为 `@pytest.mark.slow`（实际构建 wheel 较慢），默认跑 `pytest` 会跳过，需要 `pytest -m slow` 单独跑

**验收标准**
- [x] `pytest` 本地跑通，62 项普通测试全部通过，1 项 slow wheel 测试按预期跳过（含 Python 3.8 打包、Notebook 素材、Palm 检测和 discovery 多档测试）
- [x] `pytest -m slow` 单独跑 wheel 完整性测试通过（1 passed）
- [x] CI 配置文件语法正确（未在真实 CI 环境跑过，只做了本地语义核对）

---

### 任务 13：开发模型目录 API（`support_tasks`/`describe_task` 等）
**状态**：done

**为什么做**
用户想知道"这个库支持哪些任务"或"某个任务需要什么依赖、输入输出是什么"，原来只能去读 `workflow.py` 源码里的 `task_dict` 和文档字符串。有了 registry（任务 2）之后，这些信息已经是结构化数据了，应该让用户能直接查询，而不必看源码。

**怎么做**
新建 [XEdu/hub/model_discovery.py](XEdu/hub/model_discovery.py)，基于 `model_registry` 提供只读查询函数：
- `support_tasks()` / `support_models(task_name=None)`：列出任务/模型 ID
- `describe_task(task_name)`：返回任务的描述、输入输出类型、可用模型列表、依赖、是否支持自动下载
- `describe_model(model_id)`：返回单个模型的完整元数据
- `model_matrix(category=None)`：按 output_type 分组、按 latency_tier 排序的模型矩阵视图

再把这些函数从 [XEdu/hub/__init__.py](XEdu/hub/__init__.py) 重新导出，让用户可以直接 `from XEdu.hub import support_tasks` 而不必知道 `model_discovery` 这个内部模块名。同时把 `exceptions.py` 里的异常类也一并导出，方便用户 `except XEduError`。

**验收标准**
- [x] `from XEdu.hub import support_tasks, describe_task` 可用，验证过不产生循环 import
- [x] `describe_task('pose_face106')` 返回的是"不支持自动下载 + 原因"，而不是 `not_found`（这也是任务 2 里补注册记录的验证方式）
- [x] `describe_task('det_body')` 返回结构完整（description/input_type/output_type/available_models/optional_dependencies/auto_download_supported/recommended_for/tags）

---

## Phase 1：架构重构（进行中，当前最优先）

### 任务 5：引入 handler 架构，拆解 `workflow.py`
**状态**：done（检测类 A1-A4 已接管，且 `pose_body17`、音频、文本、图文匹配 handler 已作为非检测类验证样例接管）

**为什么做**
[XEdu/hub/workflow.py](XEdu/hub/workflow.py) 是一个 2233 行的单文件，`Workflow.__init__` 用一串 if/elif 处理 20+ 种任务的模型加载，`inference()` 用另一串 if/elif 分发到各个 `_xxx_infer` 方法，`format_output()` 又用第三串 if/elif 格式化输出。同一个任务的逻辑分散在文件里至少 3-4 个不连续的位置。

这不是抽象洁癖问题，是这次审计里两次真实 bug 的直接成因：
1. `pose_face106` 的下载 URL 错误——因为下载 URL 表和文件名表是两份独立维护的字典，改一处忘了同步改另一处
2. 早期版本的 `handlers/detection.py`（在动手写 handler 之前）差点把编造的 640×640/无归一化逻辑当成真实实现接入，如果不是回去对照 `workflow.py:1306` 的 `_det_infer` 原文核实，会产出数值错误但不报错的检测结果

只要模型配置和处理逻辑继续散落在同一个大文件的多处，这类"改一处漏一处"的错误会持续发生，而且新增模型的成本会越来越高。

**怎么做**
拆分成"registry 管数据 + handler 管行为"两层（registry 已在任务 2 完成），本任务负责 handler 层：

1. [XEdu/hub/handlers/base.py](XEdu/hub/handlers/base.py)：`BaseHandler` 抽象基类，统一 `__init__`（依赖检查 → 找 checkpoint → 加载模型）流程；`ONNXHandler`/`DetectionHandler`/`PoseHandler`/`EmbeddingHandler`/`ClassificationHandler` 等按任务类型的中间基类，定义 `inference()`/`format_output()` 的默认骨架
2. [XEdu/hub/handlers/detection.py](XEdu/hub/handlers/detection.py)：具体任务实现。**核心约束**：预处理/后处理的数值逻辑必须从 `workflow.py` 对应的 `_xxx_infer` 方法逐行搬过来，不能重新设计。当前 `_run_legacy_detection()` 就是把 [workflow.py:1306](XEdu/hub/workflow.py:1306) 的 `_det_infer` 闭包函数原样搬出来做成模块级函数，`det_body`/`det_coco` 共用它

**当前结果**：`Workflow` 已对 `det_body`、`det_coco`、`det_face`、`pose_body17` 创建对应 handler 并委托推理，不再走这些任务的旧 `_det_infer`/`_face_det_infer`/`_pose_infer` 分支。旧方法暂时保留，作为等价测试和未迁移任务的兼容参照。

**子任务（按顺序做，每步都要先验证再进入下一步）**

- **A1：把 `det_body` 接入 handler 调度**
  在 `Workflow.__init__` 或 `inference()` 里加一层分发：`self.task == 'det_body'` 时创建 `DetBodyHandler` 实例并委托调用，而不是走 `_det_infer`。只迁移这一个任务，先验证新旧架构能在同一个类里共存，不要一次性全迁。

- **A2：写"旧实现 vs 新 handler"等价测试**
  用一个返回固定数组的 mock ONNX session（不需要真实模型文件），同一份输入分别喂给旧的 `_det_infer` 和新的 `DetBodyHandler.inference()`，断言 `boxes`/`scores`/`classes` 完全相等。这一步的意义是把"数值对齐"从"人工核对代码"变成"自动化保证"，防止未来有人改动其中一份实现时忘了同步另一份。

- **A3：迁移 `det_coco`**
  验证带 `target_class` 过滤和多类别 `classes` 输出的路径。`DetCocoHandler` 的 `_run_legacy_detection()` 复用逻辑已经写好，这一步主要是接入 `Workflow` 分发层 + 补等价测试。

- **A4：迁移 `det_face`**
  这个任务比 A1-A3 复杂：`workflow.py` 里的 `_face_det_infer`（[workflow.py:1239](XEdu/hub/workflow.py:1239)）同时兼容旧 Haar Cascade 模型（通过 `scaleFactor`/`minNeighbors` 等旧参数触发）和新 YuNet 模型。迁移时必须把两条路径都保留，不能因为"YuNet 是新代码"就丢掉 Cascade 兼容分支——那是给已经保存了旧 `.xml` cascade 文件的用户用的。

**验收标准**
- [x] A1：`Workflow(task='det_body').inference(img)` 实际调用 `DetBodyHandler`，返回格式与迁移前一致（`tests/test_detection_handler_equivalence.py`）
- [x] A2：等价测试通过，且能在未来的 CI 里跑（mock ONNX session 对比旧 `_det_infer` 与新 handler 输出）
- [x] A3：`det_coco` 迁移完成，`target_class` 过滤行为一致（覆盖 `classes` 输出路径）
- [x] A4：`det_face` 的 Cascade 和 YuNet 两条路径都验证过
- [x] 非检测类验证：`pose_body17` 已迁移到 `PoseBody17Handler`，并用 mock session 对比旧 `_pose_infer` 输出
- [x] 全部迁移完成后，普通测试与 slow 打包测试均通过（`python3 -m pytest -q`：43 passed, 1 deselected；`python3 -m pytest -m slow -q`：1 passed）

---

### 任务 7：用 logging 替代裸 print，规范异常捕获
**状态**：in_progress（已迁移 handler 文件无裸 `print()`/裸 `except:`；`workflow.py` 的模型加载/下载提示与 NLP QA 调试输出已迁移到 logging，仍保留少量历史兼容分支）

**为什么做**
仓库里有约 125 处裸 `print()` 和 15 处裸 `except:`。裸 print 没有日志级别，用户没法在非交互环境（比如批量处理脚本、CI）里调低日志噪音；裸 `except:` 会连 `KeyboardInterrupt`/`SystemExit` 都吞掉，调试时很难定位真实错误发生的位置。

**怎么做**
这是一项跨越很多文件的存量清理，不建议一次性全做（改动面太大，回归风险高）。按"接触到就顺手改"的节奏推进：
- 每次迁移一个任务到 handler 架构（任务 5 的子任务）时，顺手把该任务相关的 print/裸 except 换成 `logging.getLogger(__name__)` 和具体异常类型
- `model_store.py` 已经是这种模式的示范（这次会话里已完成，两处 `print` 换成了 `logger.info`）
- 不要单独开一个"全项目替换 print"的大 PR，容易引入无关改动导致的回归，且难以 review

**验收标准**
- [x] 每完成一个任务 5 的子任务（A1-A4），对应的 handler 文件里没有裸 print/裸 except
- [x] `Workflow` handler 调度路径和通用模型加载提示已改用 `logging.getLogger(__name__)`
- [ ] 长期目标（不要求短期完成）：`workflow.py` 里的 print 数量随着 handler 迁移进度递减
- 不设置"全部完成"的验收标准，因为这本质是伴随架构迁移的持续性工作，不是一次性任务

---

## Phase 1.5：收尾与文档（清理技术债）

### B1：清理任务状态
**状态**：done — 本文件（TASKS.md）即为最新状态，且已同步更新 TaskUpdate 工具里的任务状态

### B2：清理 `XEdu/examples/Untitled.ipynb`
**状态**：done

**为什么做**
这是一个空的 Jupyter scratch 文件（只有 `print(1)` 之类的测试内容），从这次会话一开始就以未跟踪文件的状态存在，不属于任何功能改动，不应该进入版本库。

**怎么做**
确认内容确实无用后直接删除。如果用户有留着的理由（比如正在用它做本地实验），改成不删除但加入 `.gitignore`。

**验收标准**
- [x] 文件被删除，或被显式加入 `.gitignore` 并说明原因（已删除 scratch notebook）

---

### B3：补 CHANGELOG / 迁移说明
**状态**：done

**为什么做**
这次改动里有几处是**主动引入的破坏性变更**（不是意外破坏，是为了修复隐藏问题而故意让某些原本"能跑但结果是错的"调用方式现在报错）。如果不写清楚，用户升级后遇到报错会以为是新 bug，实际是需要调整调用方式。

**怎么做**
新建或更新 CHANGELOG，至少包含以下四条（这是本次会话逐一核实过的真实行为变更，不是推测）：

1. **移除运行时自动 `pip install`**（影响 [XEdu/LLM/client.py](XEdu/LLM/client.py)、[XEdu/LLM/llms/gradioapi.py](XEdu/LLM/llms/gradioapi.py)、[XEdu/__init__.py](XEdu/__init__.py) 的节日彩蛋）。缺少 `gradio`/`gradio-client`/`holidays` 时，之前会自动执行 `pip install` 再重试，现在直接抛 `ImportError` 并提示手动安装命令。**受影响用户**：依赖"首次调用自动装好依赖"这个行为、且环境没有预装这些包的用户。
2. **`pose_face106` 不再静默下载错误模型**。本地无 checkpoint 且未指定 `checkpoint=` 参数时，之前会静默下载 `pose_wholebody133` 的权重文件当作 face106 使用（结果是错的但不报错），现在直接抛 `RuntimeError` 并提示手动指定 `checkpoint=`。**受影响用户**：任何用过 `pose_face106` 默认下载路径的人——需要说明的是，他们之前得到的结果本来就是错的，这个变更是让错误可见，不是制造新问题。
3. **`RepoModel._custom_infer` 参数类型变化**（[XEdu/hub/repo_model.py](XEdu/hub/repo_model.py)）。之前通过 `eval()` 拼接字符串调用自定义推理函数，所有 kwarg 值会被强制转成字符串（例如 `threshold=0.5` 实际传入的是字符串 `"0.5"`）；现在直接用 `**kwargs` 传递，保留原始类型。**受影响用户**：写了 `data_process.py` 自定义推理逻辑、且这些逻辑依赖"收到的参数都是字符串"这一旧行为的用户，需要检查并调整类型处理代码。
4. **打包配置修复**（`pyproject.toml`）。修复了 `XEdu.LLM`/`XEdu.utils`/`XEdu.hub.models`/`XEdu.hub.BaseDT` 四个子包曾被打包遗漏的问题（旧配置 `packages=["XEdu"]` 不会递归发现子包）。**受影响用户**：无负面影响，纯粹是修复——旧版本正式 `pip install` 后这些模块本来就是缺失的。

**验收标准**
- [x] CHANGELOG 文件存在，覆盖以上四条，每条都说明"之前行为 / 现在行为 / 受影响用户"
- [x] 版本号按语义化版本规则递增（`pyproject.toml` 与 `XEdu/version.py` 均更新为 `3.0.0`）

---

## Phase 2：模型生态扩展（进行中）

这是最初目标"让模型生态更丰富"的主体部分。**前提条件 Phase 1（任务 5，handler 架构真正接管 `Workflow` 调度）已完成。** 当前已从任务 8/10/11 开始推进，后续新增能力必须继续遵守"真实模型资源、不编造 URL、先能被 `Workflow` 调到"的原则。

### 任务 8：扩充视觉检测模型矩阵（多档 + 新类别）
**状态**：done（先完成多档矩阵；新类别仍需真实模型资源后再追加）

**为什么做**
当前检测类任务大多只有一个默认模型（比如 `det_body` 只有一档），用户没有"轻量但快 / 精度高但慢"之间的选择。教学场景里，树莓派/教室电脑的算力差异很大，单一档位模型没法兼顾。

**怎么做**（等任务 5 完成后再启动）
1. 在 `model_registry.py` 里为同一个 `task_name` 注册多个 `model_id`（架构已支持——`get_models_by_task()` 本来就返回列表），用 `latency_tier`/`quality_tier` 字段区分档位
2. 每个新模型对应的 handler 复用同一个 `task_name` 的处理逻辑（如果输入输出格式相同），或者在 handler 里区分 `model_id` 做不同的前后处理
3. 通过 `model_discovery.describe_task()` 让用户能看到"这个任务有哪些档位可选"

**已完成**
- `det_body` 现在包含 `det_body-default`（base）和 `det_body_l-default`（large/high）两档；保留 `det_body_l` 旧任务名兼容入口 `det_body_l-compat`
- `det_coco` 现在包含 `det_coco-default`（base）和 `det_coco_l-default`（large/high）两档；保留 `det_coco_l` 旧任务名兼容入口 `det_coco_l-compat`
- `Workflow(task="det_body", model_id="det_body_l-default")` 可选择指定档位
- `describe_task()` 新增 `models` 明细字段，展示每个模型的 latency/quality/recommended_for 等信息

**验收标准**
- [x] 至少一个检测任务有 2 个及以上档位可选（`det_body` 与 `det_coco` 均已满足）
- [x] `describe_task()` 能正确展示多档位信息（`tests/test_model_discovery.py`）
- [x] 新增模型的真实下载 URL 已核实（复用原 `det_body_l`/`det_coco_l` 已核实 URL，没有编造新 URL）

---

### 任务 9：扩充人脸关键点模型矩阵（不再强绑定 106 点）
**状态**：done（新增 `pose_face_landmark` 双档模型；旧 `pose_face106` 继续保持禁用兼容入口）

**为什么做**
`pose_face106` 目前处于"能查询到、但不能自动下载"的状态（任务 2 已经如实标注了这一点）。这不是最终目标状态，只是"总比静默下载错误模型好"的临时止损。用户已明确核心诉求是"人脸关键点检测"，不要求必须是 106 点，但也明确不接受 MediaPipe 这类新增重依赖方案。因此后续不要被旧 `face106.onnx` 约束，也不要把默认路线切到新的重运行栈；应把能力拆成：
- `pose_face_landmark`：推荐的新默认人脸关键点任务，优先轻量 ONNX、无新增运行时依赖
- `pose_face106`：保留为 106 点兼容入口，只有许可清晰且验证过的模型才开启自动下载

**怎么做**
1. 首选轻量 ONNX 方案，继续复用项目现有 `onnxruntime`/OpenCV 依赖；默认任务不能新增 `mediapipe`、`modelscope`、`torch` 等运行时依赖
2. 优先把 ModelScope/IIC MobileNet 106 点模型导出为 ONNX 并托管到当前 `wht0926/xedu-hub-models`，它许可清晰、体积小、可复用现有 YuNet/`det_face` 人脸检测
3. 可追加 PIPNet/WFLW-98 作为非 106 点轻量 ONNX 候选，但必须先确认模型权重许可、真实 ONNX 文件和端到端效果
4. 保留旧 `pose_face106-default` 的禁用状态，避免继续传播质量不佳或来源不明的旧 `face106.onnx`
5. `pose_face_landmark` handler 需要包含人脸检测、裁剪、关键点模型推理和坐标反变换；当前先采用 YuNet 检测框 + padding 裁剪，后续如需继续追求官方 pipeline 对齐，再补旋转/迭代细化
6. **绝对不要在拿不到真实地址时编造一个 UUID 去填**——这正是本次审计中发现并纠正过的错误做法，编造的 URL 会 404，比显式报错更难排查

**当前核查结果**
- XEdu 官方预置任务文档确认 `pose_face106` 是人脸 106 点任务，也说明未指定 checkpoint 时会通过网络下载，但页面没有公开具体 `openinnolab` asset ID：https://xedu.readthedocs.io/zh-cn/master/xedu_hub/preset_task.html
- OpenInnoLab 常见模型合集页面是模型下载入口页，但公开页面没有给出可直接写入代码的 `face106.onnx` URL：https://www.openinnolab.org.cn/pjedu/courses/courseDetail?courseId=6684e63a545bd744a5d923f8
- 推荐候选：ModelScope/IIC `cv_mobilenet_face-2d-keypoints_alignment`，Apache License 2.0，MobileNet 106 点人脸关键点模型，官方说明输出 106 点关键点与 pitch/roll/yaw 姿态角，输入大小 96x96，模型权重约 3.26MB：https://modelscope.cn/models/iic/cv_mobilenet_face-2d-keypoints_alignment
- 不采用为默认：MediaPipe Face Landmarker 效果强，但新增 `mediapipe` 运行时依赖且不是 ONNX registry 路线；当前项目默认路线应保持轻量 ONNX
- 已下载候选模型并读取 `configuration.json`/README；ModelScope/IIC 原仓库发布的是 `pytorch_model.pt`，不是现成 ONNX
- 已按 EasyCV 源码复刻模型本体并导出单文件 ONNX：`face_landmark106_mobilenet.onnx`，sha256 `8b730ef412d8db18e6af5602fd2bc052f5afba3d330de5be8fb398ab9a2f6cff`
- 已追加 PIPNet/WFLW-98 高精度档：`face_landmark98_pipnet_wflw.onnx`，sha256 `9862838dc6144bc772b6485f6f6d31295c0b1c1ab7293e6ddeb0a439cb10218d`，GitHub release 作为备用源
- 两个 ONNX 均已上传到 ModelScope `wht0926/xedu-hub-models`，运行时仍只通过直接文件 URL 下载，不新增 `modelscope` SDK 运行时依赖
- 已新增 `PoseFaceLandmarkHandler`：复用 YuNet 检测人脸，默认输出 106 点，可通过 `model_id="pose_face_landmark-pipnet98-wflw"` 切换 98 点高精度档
- InsightFace `2d106det.onnx` 也很成熟，但预训练模型包需要单独商业授权，不适合作为 XEdu 默认公开模型源；可作为用户自带 checkpoint 的兼容参考

**验收标准**
- [x] 新增不强绑定点数的 `pose_face_landmark` 任务，README 明确说明它与 `pose_face106` 的区别
- [x] 默认实现只使用现有轻量依赖：OpenCV + onnxruntime + numpy，不引入 MediaPipe/Torch/ModelScope 作为运行时依赖
- [x] `model_registry.py` 中新增 `pose_face_landmark-mobilenet106` 默认档和 `pose_face_landmark-pipnet98-wflw` 高精度档，source 指向 ModelScope 主源并记录 SHA256
- [x] 新增 `PoseFaceLandmarkHandler`，包含 YuNet 人脸检测、裁剪、模型推理、PIPNet/WFLW-98 解码和坐标反变换
- [x] 真实图片端到端验证：默认 MobileNet 106 与 PIPNet/WFLW-98 均可自动下载并输出人脸关键点
- [x] 保留旧 `pose_face106-default` 禁用说明，不让用户误下载旧错误/低质量模型

---

### 任务 10：建立音频模型生态（分类 + 关键词 + embedding）
**状态**：done

**为什么做**
当前音频相关能力只有 `embedding_audio`（CLAP）一个孤立任务，[XEdu/hub/models/clap.py](XEdu/hub/models/clap.py) 是唯一的音频模型实现，没有分类、关键词检测等更贴近课堂"感知 AI"实验的能力。

**怎么做**
1. 先定义统一的音频输入协议（文件路径 / numpy waveform / sample rate 处理方式），避免每个音频任务各自处理格式
2. 新增 `handlers/audio.py`，参照 `EmbeddingHandler` 的模式扩展出 `AudioClassificationHandler` 等
3. 优先选择"轻量、能在 CPU 上跑、适合课堂演示"的模型，而不是追求 SOTA 精度

**验收标准**
- [x] 至少新增 1 个音频分类或关键词检测任务（`cls_audio`，基于 CLAP embedding 与用户提供的参考原型做余弦相似度分类）
- [x] 关键词检测任务已新增（`det_audio_keyword`，基于 CLAP embedding 与关键词/声音事件参考原型做阈值检测）
- [x] 音频任务的输出格式和视觉任务风格一致（`embedding_audio`/`cls_audio`/`det_audio_keyword` 均支持 `format_output(lang=...)`）
- [x] 统一音频输入协议已扩展为文件路径 / numpy waveform / `(waveform, sample_rate)` / dict 输入
- [x] 真实音频文件 + 真实 CLAP 权重端到端人工验证（`embedding_audio` 输出 `(1, 1024)`，`cls_audio` 与 `det_audio_keyword` 对本地合成 wav 均命中 `tone440`）

**验证备注**
- Python `requests` 访问 `https://www.openinnolab.org.cn/...embedding_audio.onnx` 时遇到站点证书过期导致的 SSL 校验失败；本次人工验证使用 `curl -k -L -C -` 临时下载到 `/tmp/xedu_model_verify/embedding_audio.onnx`，不把跳过证书校验写入库代码。

---

### 任务 11：建立轻量 NLP 模型线（离线文本分类 + 中文 embedding）
**状态**：done（已完成离线原型文本分类；中文最小验证完成，但当前模型不是中文专用 embedding）

**为什么做**
当前 NLP 能力分裂成两个不相关的世界：`XEdu.hub` 里只有 `nlp_qa`（基于 SQuAD 的问答，用的是 2018 年的 BERT tokenizer 代码）；`XEdu.LLM` 是调用外部大模型 API 的 wrapper。中间缺一层"轻量、离线、传统 NLP"的能力——教学场景里"不联网也能跑"的价值往往被低估。

**怎么做**
1. 新增 `handlers/nlp.py`，扩展出文本分类、情感分析等基于小模型（不是 LLM）的 handler
2. 中文文本 embedding 可以参照 `embedding_text`（CLIP 文本侧）的模式，但需要确认现有 CLIP 模型对中文的支持程度，可能需要引入专门的中文 embedding 模型
3. 明确在文档里区分"这是离线小模型"还是"这是调用在线 LLM API"，避免用户混淆 `XEdu.hub` 和 `XEdu.LLM` 的适用场景

**验收标准**
- [x] 至少新增 1 个离线文本分类任务（`cls_text`，基于本地 `embedding_text`/CLIP 文本 embedding 与参考原型分类）
- [x] 文档明确说明该任务不需要网络/API key，与 `XEdu.LLM` 的使用场景区分开（`README.md` 与 `model_registry.py` 中均标注为本地离线能力）
- [x] 中文 embedding 支持质量已做最小人工验证：中文样例能按预期分到“人工智能/体育”，但类别相似度差距较小，说明当前 CLIP 文本侧不是可靠的中文专用 embedding；后续如果要做正式中文语义任务，仍建议引入专门中文 embedding 模型

---

### 任务 12：建立多模态任务能力（图文匹配 + 简单 VQA）
**状态**：in_progress（图文匹配已完成；简单 VQA 仍需合适模型资源）

**为什么做**
现有 `embedding_image`/`embedding_text` 已经是 CLIP 的图像侧和文本侧，具备做图文匹配的基础，但没有包装成一个直接可用的任务。多模态能力对应"更丰富的模型生态"里最有教学特色的部分（比如"找出和这句话最匹配的图片"），值得在基础打牢后重点投入。

**怎么做**
1. 依赖任务 11 先把文本处理链路理清楚，避免图文匹配任务里的文本处理逻辑又重新发明一套
2. 新增 `handlers/multimodal.py`，基于已有的 `embedding_image`/`embedding_text` 模型组合出图文相似度计算（可以复用 [XEdu/utils/utils.py](XEdu/utils/utils.py) 里已有的 `get_similarity()` 函数）
3. 简单 VQA（视觉问答）需要额外的模型支持，如果没有合适的轻量开源模型，可以先只做图文匹配，VQA 作为后续目标

**已完成**
- 新增 [XEdu/hub/handlers/multimodal.py](XEdu/hub/handlers/multimodal.py)，实现 `match_image_text`，复用 `embedding_image`/`embedding_text` 两个本地模型计算 CLIP 相似度
- `Workflow(task="match_image_text")` 已接入 handler 分发，支持图片路径/数组、候选文本、预计算 image/text embeddings
- `model_registry.py` 已注册 `match_image_text-clip` 组合模型，明确这是 composite 任务，不编造新的模型 URL
- `README.md` 已说明 `match_image_text` 是图文匹配，不是 VQA；简单 VQA 需要额外视觉问答模型，当前未实现

**验收标准**
- [x] 图文匹配任务端到端跑通（给一张图和几句候选文本，返回最匹配的文本及相似度分数；`tests/test_multimodal_handlers.py`）
- [x] 明确说明 VQA 能力的现状（当前未实现，依赖后续找到合适的轻量 VQA 模型）

---

## 下一步执行顺序（推荐）

1. **A1 → A2 → A3 → A4**（任务 5 的四个子任务）：把检测类任务真正迁移到 handler 架构，每步都先写等价测试再继续
2. **B2 → B3**：清理 scratch 文件，把破坏性变更写成 CHANGELOG——这一步可以和 A1-A4 并行，不冲突
3. ~~用同样的模式迁移至少一个非检测类任务（比如 `pose_body17`），验证 handler 架构对姿态类任务同样适用，不是"只对检测类好用"~~（已完成：`pose_body17`）
4. 开始 Phase 2，从任务 8（视觉检测矩阵）启动，因为它最贴近已有代码，风险最低

## 验收原则（写给未来接手者）

- 不要因为"文件已创建"就标记完成，必须能 import、能跑 pytest、或能手动验证行为——这条原则本身就是这次审计的直接教训
- 新 handler 的数值逻辑必须与旧 `workflow.py` 对应方法逐位对齐，不能凭空重新设计预处理/后处理逻辑；如果不确定，去读旧代码原文，不要凭经验猜
- 每次迁移一个任务后，先写等价测试，再考虑是否删除旧分支，避免出现"两份逻辑同时存在但会分叉"的中间状态长期停留
- 涉及模型下载 URL 时，绝不编造 UUID 或路径；找不到真实来源时，宁可显式报错也不要假装修复——这是本次审计中真实犯过、又真实纠正过的错误，值得反复提醒
