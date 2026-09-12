# -*- coding: utf-8 -*-
"""随时汇总 exp_koopman_gate 的（可以是未跑完的）结果。

用法：python summarize_gate.py [tag]
主实验每跑完一个被试就落一次盘，所以中途也能看。
"""
import json
import os
import sys

from config import RESULTS_DIR
from exp_koopman_gate import summarize

tag = sys.argv[1] if len(sys.argv) > 1 else "_v2"
path = os.path.join(RESULTS_DIR, "koopman_gate", f"gate_rows{tag}.json")
rows = json.load(open(path, encoding="utf-8"))
subs = sorted({r["subject"] for r in rows})
frs = sorted({r.get("framing") for r in rows if r.get("framing")})
print(f"{path}\n已完成被试：{subs}（{len(subs)} 个）  帧化条件：{frs}")
summarize(rows, frs)
