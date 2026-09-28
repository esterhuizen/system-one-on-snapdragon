"""DirectML rejects Reshape(allowzero=1) (emitted by torch's dynamo ONNX exporter). Rewrite to allowzero=0.

Only the graph proto is rewritten; external weight files are referenced in place (write next to the source).
python fix_allowzero.py in.onnx out.onnx
"""
import sys
import onnx
from onnx import numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
consts = {i.name: i for i in m.graph.initializer if i.data_location != onnx.TensorProto.EXTERNAL}
for n in m.graph.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                consts[n.output[0]] = a.t
changed = unsafe = 0
for n in m.graph.node:
    if n.op_type != "Reshape":
        continue
    for a in list(n.attribute):
        if a.name == "allowzero" and a.i == 1:
            shp = consts.get(n.input[1])
            if shp is not None and 0 in numpy_helper.to_array(shp).tolist():
                unsafe += 1
                continue
            n.attribute.remove(a)
            changed += 1
onnx.save(m, dst)
print(f"Reshape allowzero=1 -> 0: {changed} rewritten, {unsafe} left (shape contains 0)")
