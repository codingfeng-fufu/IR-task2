# Utils - 工具脚本

**用途**：辅助工具和修复脚本
**特点**：项目维护和问题修复

## 📁 文件列表

| 文件 | 行数 | 功能 | 使用场景 |
|------|------|------|----------|
| `fix_evaluator.py` | 14 | 修复evaluator.py格式问题 | 格式化错误修复 |

## 🔧 工具说明

### fix_evaluator.py - 评估器修复工具

**问题背景**：
- `evaluator.py` 在打印混淆矩阵时可能出现字符串格式化错误
- 特殊字符转义问题导致输出异常

**解决方案**：
- 自动替换问题代码行
- 修复字符串转义问题
- 确保混淆矩阵正常打印

**使用方法**：
```bash
cd /home/u2023312337/task2/task2
python fix_evaluator.py
```

**输出**：
```
✓ evaluator.py 已修复
```

**修复内容**：
```python
# 问题代码（修复前）
print(f"{'实际\\预测':<15} {'预测为负':<15} {'预测为正':<15}")

# 修复后代码
header = "实际\\预测"
print(f"{header:<15} {'预测为负':<15} {'预测为正':<15}")
```

**注意事项**：
- ⚠️ 会直接修改 `evaluator.py` 文件
- ⚠️ 运行前建议备份
- ✅ 修复是幂等的（可重复运行）

## 💡 工具设计理念

### 为什么需要独立的修复脚本？

**传统方式（手动修复）**：
```bash
vim evaluator.py
# 手动查找问题行
# 手动修改代码
# 保存退出
```

**脚本方式（自动修复）**：
```bash
python fix_evaluator.py
# 一键完成
```

**优势**：
- ⚡ 快速修复
- 🔧 自动化
- ✅ 准确无误
- 📝 可追溯

### 适用场景

1. **开发阶段**：
   - 发现格式化问题
   - 快速修复并继续开发

2. **部署阶段**：
   - 环境迁移后出现问题
   - 自动修复脚本恢复正常

3. **维护阶段**：
   - 代码重构后的快速修复
   - 批量修复类似问题

## 🔍 技术细节

### 字符串替换逻辑

```python
# 1. 读取文件
with open('evaluator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 2. 精确替换
content = content.replace(
    "旧代码（问题行）",
    "新代码（修复后）"
)

# 3. 写回文件
with open('evaluator.py', 'w', encoding='utf-8') as f:
    f.write(content)
```

**特点**：
- 精确匹配（避免误修改）
- UTF-8编码（支持中文）
- 完整文件读写（保持格式）

### 扩展性

**添加新的修复规则**：
```python
# 在 fix_evaluator.py 中添加
content = content.replace(
    "问题代码2",
    "修复代码2"
)

content = content.replace(
    "问题代码3",
    "修复代码3"
)
```

**创建新的修复脚本**：
```python
# fix_xxx.py
with open('xxx.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("问题", "修复")

with open('xxx.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ xxx.py 已修复")
```

## 📚 相关文档

- **Stage1_Foundation/README.md** - evaluator.py的功能说明
- **Main_Scripts/README.md** - 主流水线使用说明

## 🔗 后续扩展

### 可能的工具脚本

1. **fix_chinese_encoding.py**
   - 修复中文编码问题
   - 统一UTF-8编码

2. **update_paths.py**
   - 批量更新文件路径
   - 环境迁移工具

3. **cleanup_outputs.py**
   - 清理输出目录
   - 删除临时文件

4. **validate_models.py**
   - 验证模型文件完整性
   - 检查模型可加载性

5. **migrate_config.py**
   - 配置文件版本迁移
   - 向后兼容处理

---

**总结**：工具脚本目录提供辅助功能，简化项目维护和问题修复。
