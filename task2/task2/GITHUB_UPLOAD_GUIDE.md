# GitHub上传指南

本文档提供两种方式上传Task2项目到GitHub:
1. **自动方式**: 运行准备好的脚本
2. **手动方式**: 逐步执行命令

## 📦 上传内容说明

### ✅ 将要上传的文件

**代码文件** (~10,000行):
- 所有Python源代码 (*.py)
- 6个阶段的完整实现 (stages/目录)
- 主程序和工具脚本

**文档文件** (~5,600行):
- README.md (中文项目介绍)
- CLAUDE.md (使用指南)
- VERSION_EVOLUTION.md (技术演进)
- EVOLUTION_ROADMAP.md (可视化路线图)
- LLM_EXPERIMENT_GUIDE.md (LLM实验指南)
- presentation_docs/ (14个演示文档)

**配置文件**:
- requirements.txt
- llm_config_template.json
- .gitignore
- .gitattributes

**目录结构**:
- .gitkeep文件 (保留data/, models/, output/等目录)

**输出结果** (文本文件和可视化图片):
- output/evaluation_results.txt
- output/all_stages_comparison.json
- output/all_stages_comparison_report.txt
- output/*.png (模型对比图、混淆矩阵、t-SNE可视化)
- output/bert_experiments/*.png (BERT实验结果可视化)
- output/llm_experiments/*.png (LLM实验结果可视化)

### ❌ 不会上传的文件 (已在.gitignore中排除)

**数据文件** (~15MB):
- data/positive.txt
- data/negative.txt
- data/testSet-1000.xlsx

**模型文件** (~1.5GB):
- models/*.pt (BERT模型)
- models/*.pkl (NB和SVM模型)
- models/*.model (Word2Vec模型)
- models/experiments/ (实验模型)


**系统文件**:
- .venv/ (虚拟环境)
- __pycache__/ (Python缓存)
- *.log (日志文件)
- checkpoints/ (训练检查点)

## 方式1: 自动上传 (推荐)

### 步骤1: 使脚本可执行

```bash
cd /home/u2023312337/task2/task2
chmod +x prepare_github_upload.sh
```

### 步骤2: 运行脚本

```bash
./prepare_github_upload.sh
```

脚本会:
1. 显示当前git状态
2. 移除已跟踪的大文件
3. 添加所有代码和文档
4. 显示将要提交的内容
5. 提交更改
6. 推送到GitHub

每个关键步骤都会提示确认,可以随时取消。

### 步骤3: 验证上传

访问 https://github.com/codingfeng-fufu/IR-task2 确认:
- README.md正确显示
- 文件结构完整
- 代码和文档都已上传

## 方式2: 手动上传 (逐步执行)

如果你希望更细粒度地控制每一步,可以手动执行以下命令:

### 步骤1: 检查当前状态

```bash
cd /home/u2023312337/task2/task2
git status
```

### 步骤2: 从git跟踪中移除大文件

**重要**: 这些命令只是移除git跟踪,不会删除本地文件!

```bash
# 移除模型文件
git rm --cached models/best_bert_model.pt
git rm --cached models/naive_bayes_model.pkl
git rm --cached models/word2vec_svm_model_svm.pkl
git rm --cached models/word2vec_svm_model_w2v.model

# 移除output中的大文件
git rm --cached output/confusion_matrices.png
git rm --cached output/model_comparison.png
git rm --cached output/predictions.json
git rm --cached output/tsne_*.png
```

### 步骤3: 添加所有代码和文档

```bash
# 添加当前目录的Python文件
git add *.py

# 添加文档
git add *.md
git add *.json
git add *.txt

# 添加配置
git add .gitignore
git add .gitattributes

# 添加各阶段代码
git add stages/

# 添加演示文档包
git add presentation_docs/

# 添加目录结构保留文件
git add data/.gitkeep
git add models/.gitkeep
git add models/experiments/.gitkeep
git add output/.gitkeep

# 添加output中的结果文件
git add output/evaluation_results.txt
git add output/all_stages_comparison.json
git add output/all_stages_comparison_report.txt

# 添加所有可视化图片 (均 < 1MB)
git add output/*.png
git add output/bert_experiments/*.png
git add output/llm_experiments/*.png
```

### 步骤4: 查看将要提交的内容

```bash
# 查看状态
git status --short

# 查看详细变更
git diff --cached --stat

# 统计文件数
echo "Python文件: $(git diff --cached --name-only | grep '\.py$' | wc -l)"
echo "文档文件: $(git diff --cached --name-only | grep '\.md$' | wc -l)"
echo "总文件数: $(git diff --cached --name-only | wc -l)"
```

### 步骤5: 提交更改

```bash
git commit -m "准备GitHub发布: 添加完整文档和代码,包含可视化结果

- 添加中文README.md (项目介绍)
- 添加完整的6阶段代码实现
- 添加presentation_docs文档包 (14个文件)
- 添加所有技术文档 (CLAUDE.md, VERSION_EVOLUTION.md等)
- 添加output可视化图片 (16张PNG, 共~8MB)
- 移除大型模型文件 (*.pt, *.pkl, *.model)
- 移除数据文件 (*.txt, *.xlsx)
- 保留目录结构 (.gitkeep文件)

核心成果:
- 最高准确率: 90.47% (Kimi-K2 LLM)
- BERT优化: 89.04%
- 代码规模: 10,000+行
- 文档规模: 5,600+行"
```

### 步骤6: 配置远程仓库

```bash
# 检查是否已配置
git remote -v

# 如果没有配置,添加远程仓库
git remote add origin https://github.com/codingfeng-fufu/IR-task2.git

# 验证配置
git remote -v
```

### 步骤7: 推送到GitHub

```bash
# 推送到master分支
git push -u origin master
```

如果推送失败,可能需要:

**选项A: 使用HTTPS (需要GitHub个人访问令牌)**
```bash
# 1. 在GitHub生成个人访问令牌 (Settings → Developer settings → Personal access tokens)
# 2. 推送时输入用户名和令牌
git push -u origin master
# Username: 你的GitHub用户名
# Password: 你的个人访问令牌 (不是密码!)
```

**选项B: 使用SSH (需要配置SSH密钥)**
```bash
# 1. 检查是否有SSH密钥
ls ~/.ssh/id_rsa.pub

# 2. 如果没有,生成SSH密钥
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 3. 添加公钥到GitHub (Settings → SSH and GPG keys)
cat ~/.ssh/id_rsa.pub

# 4. 更改远程仓库URL为SSH
git remote set-url origin git@github.com:codingfeng-fufu/IR-task2.git

# 5. 再次推送
git push -u origin master
```

**选项C: 强制推送 (如果远程已有不同历史)**
```bash
# ⚠️ 警告: 这会覆盖远程仓库的历史!
git push -u origin master --force
```

### 步骤8: 验证上传

访问你的仓库: https://github.com/codingfeng-fufu/IR-task2

检查:
- ✅ README.md显示正确
- ✅ 代码文件都已上传
- ✅ 文档目录完整
- ✅ 大文件未上传

## 🔧 故障排查

### 问题1: 推送被拒绝 (rejected)

**原因**: 远程仓库有本地没有的提交

**解决**:
```bash
# 选项A: 拉取并合并
git pull origin master --allow-unrelated-histories
git push -u origin master

# 选项B: 强制推送 (会丢失远程的提交!)
git push -u origin master --force
```

### 问题2: 认证失败

**原因**: 没有配置GitHub凭据

**解决**: 使用上面的选项A (HTTPS + 令牌) 或选项B (SSH密钥)

### 问题3: 文件太大无法上传

**原因**: 某些文件超过GitHub限制 (100MB)

**解决**:
```bash
# 检查大文件
find . -type f -size +50M

# 移除大文件
git rm --cached 大文件路径

# 重新提交
git commit --amend
git push -u origin master
```

### 问题4: 想要撤销所有更改

```bash
# 撤销暂存 (git add)
git reset HEAD

# 撤销提交 (git commit)
git reset --soft HEAD~1

# 完全重置到上一个提交
git reset --hard HEAD
```

## 📊 上传后的仓库大小估算

**预计大小**: ~13-15 MB

包含:
- Python代码: ~2 MB
- 文档: ~2 MB
- 配置和脚本: ~500 KB
- 可视化图片: ~8 MB (16张PNG)
- 结果文件: ~500 KB

不包含:
- 数据文件: ~15 MB (已排除)
- 模型文件: ~1.5 GB (已排除)
- 大型输出: ~100 MB (已排除)

## ✅ 上传完成后的建议

### 1. 优化GitHub仓库展示

在GitHub仓库页面:
- **About**: 添加项目描述
  ```
  学术标题分类系统 | 90.47%准确率 | BERT优化 | LLM零训练 | 完整6阶段实现
  ```

- **Topics**: 添加标签
  ```
  machine-learning, nlp, bert, text-classification,
  deep-learning, scikit-learn, pytorch, llm,
  feature-engineering, academic-paper
  ```

- **Website**: 如果有演示网站,添加链接

### 2. 创建Release

```bash
# 创建标签
git tag -a v1.0 -m "初始发布版本

核心成果:
- 最高准确率: 90.47% (Kimi-K2)
- BERT优化: 89.04%
- 完整6阶段实现
- 10,000+行代码
- 5,600+行文档"

# 推送标签
git push origin v1.0
```

然后在GitHub上创建Release,添加:
- 标题: "Task2 v1.0 - 学术标题分类系统"
- 描述: 复制README.md的核心成果部分
- 附件: 可以上传演示视频或PPT

### 3. 添加徽章 (Badges)

在README.md顶部添加:
```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.12+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-completed-brightgreen.svg)
```

### 4. 考虑添加

- **CONTRIBUTING.md**: 如果接受贡献
- **LICENSE**: 开源许可证文件
- **CHANGELOG.md**: 版本更新日志
- **.github/workflows**: CI/CD配置

## 📞 需要帮助?

如果遇到问题:
1. 检查本文档的「故障排查」部分
2. 运行 `git status` 查看当前状态
3. 使用 `git log --oneline -5` 查看最近提交
4. 查看GitHub文档: https://docs.github.com/

---

**文档创建**: 2025-12-09
**目标仓库**: https://github.com/codingfeng-fufu/IR-task2
**分支**: master
