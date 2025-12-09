with open('evaluator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换问题行
content = content.replace(
    "print(f\"{'实际\\\\预测':<15} {'预测为负':<15} {'预测为正':<15}\")",
    'header = "实际\\\\预测"\n        print(f"{header:<15} {\'预测为负\':<15} {\'预测为正\':<15}")'
)

with open('evaluator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ evaluator.py 已修复")
