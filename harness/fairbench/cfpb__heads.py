import sys, json
sys.path.insert(0, os.path.expandvars("$WIN_HOME/local-laya/results/fairbench-sources/pylib"))
sys.path.insert(0, os.path.expandvars("$WIN_HOME/local-laya/src/laya-snapdragon"))
import os
import types; sys.modules['onnxruntime'] = types.ModuleType('onnxruntime')
from laya_snapdragon.common import build_prefix
from laya_snapdragon.tokenizer import Tokenizer
tok = Tokenizer(os.path.expandvars("$WIN_HOME/local-laya/models/npu/laya/tokenizer"))
A1 = {"credit_reporting": "Credit reports, credit scores, or a credit bureau",
      "debt_collection": "A debt collector or the collection of a debt",
      "credit_card": "A credit card or prepaid card",
      "bank_account": "A checking or savings account",
      "money_transfer": "Sending or receiving money, payment apps, or virtual currency",
      "mortgage": "A home mortgage or home equity loan",
      "consumer_loan": "A vehicle, student, payday, or personal loan"}
YES = "the debt is not theirs, was already paid, came from identity theft, or was discharged"
NO = "the complaint is about something else, such as collection tactics, notices, or threats"
qs = {"product": {"t": "choice", "ins": "Which financial product is this complaint about?", "crit": A1},
      "not_owed": {"t": "noul", "ins": "Does the consumer say they do not owe the debt that is being collected?", "crit": {"true": YES, "false": NO}},
      "not_owed_ab": {"t": "choice", "ins": "Does the consumer say they do not owe the debt that is being collected?", "crit": {"A": YES, "B": NO}}}
for k, q in qs.items():
    ids, m = build_prefix(tok, q, 192); full, _ = build_prefix(tok, q, 10000)
    print(k, len(ids), len(full), len(m))
