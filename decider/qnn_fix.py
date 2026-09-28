"""Make the dynamo-exported ModernBERT encoder QNN-GPU friendly (graph proto only; weights referenced in place):
  1) drop SDPA's NaN guard  Where(IsNaN(softmax), 0, softmax) -> softmax   (IsNaN fails QNN GPU finalize)
  2) fuse erf-GELU (Div->Erf->Add->Mul) into com.microsoft Gelu            (Erf is not supported by QNN)
python qnn_fix.py in.onnx out.onnx
"""
import sys
import onnx
from onnx import helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input:
        cons.setdefault(i, []).append(n)

def rename_input(old, new):
    for n in g.node:
        for k, i in enumerate(n.input):
            if i == old:
                n.input[k] = new
    for o in g.output:
        if o.name == old:
            raise RuntimeError("graph output rename not handled")

remove = set()
# 1) NaN guard
nan_fixed = 0
for n in list(g.node):
    if n.op_type == "IsNaN":
        x = n.input[0]
        for w in cons.get(n.output[0], []):
            if w.op_type == "Where" and w.input[0] == n.output[0] and w.input[2] == x:
                rename_input(w.output[0], x)
                remove.update([id(n), id(w)]); nan_fixed += 1
# 2) GELU: Div(x, c) -> Erf -> Add(., 1) -> Mul(a, .) where a = Mul(x, 0.5) (either operand order)
gelu_fixed = 0
new_nodes = []
for n in list(g.node):
    if n.op_type != "Erf":
        continue
    div = prod.get(n.input[0])
    if div is None or div.op_type != "Div":
        continue
    x = div.input[0]
    adds = cons.get(n.output[0], [])
    if len(adds) != 1 or adds[0].op_type != "Add":
        continue
    add = adds[0]
    muls = cons.get(add.output[0], [])
    if len(muls) != 1 or muls[0].op_type != "Mul":
        continue
    mul = muls[0]
    other = mul.input[0] if mul.input[1] == add.output[0] else mul.input[1]
    half = prod.get(other)
    if half is not None and half.op_type == "Mul" and x in half.input:
        # form A: Mul(Mul(x, 0.5), 1 + erf)
        gelu = helper.make_node("Gelu", [x], [mul.output[0]], name=n.name + "_fused_gelu", domain="com.microsoft")
        new_nodes.append((mul, gelu))
        remove.update([id(div), id(n), id(add), id(mul)])
        if len(cons.get(half.output[0], [])) == 1:
            remove.add(id(half))
        gelu_fixed += 1
        continue
    # form B: Mul(x, Mul(0.5, 1 + erf))   (dynamo decomposition of F.gelu)
    mul2s = cons.get(mul.output[0], [])
    if len(mul2s) != 1 or mul2s[0].op_type != "Mul":
        continue
    mul2 = mul2s[0]
    if x not in mul2.input:
        continue
    gelu = helper.make_node("Gelu", [x], [mul2.output[0]], name=n.name + "_fused_gelu", domain="com.microsoft")
    new_nodes.append((mul2, gelu))
    remove.update([id(div), id(n), id(add), id(mul)])
    gelu_fixed += 1
kept = []
repl = {id(mul): gelu for mul, gelu in new_nodes}
for n in g.node:
    if id(n) in repl:
        kept.append(repl[id(n)])
    elif id(n) not in remove:
        kept.append(n)
del g.node[:]
g.node.extend(kept)
if gelu_fixed and not any(o.domain == "com.microsoft" for o in m.opset_import):
    m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
onnx.save(m, dst)
print(f"NaN guards removed: {nan_fixed}; erf-GELU fused: {gelu_fixed}; nodes now {len(g.node)}")
