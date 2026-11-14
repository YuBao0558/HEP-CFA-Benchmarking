#!/usr/bin/env python3
"""批量改造 plot/*.R 脚本: 注入 BBBD_EXPERIMENT 路径前缀 (data/experimentX, figures/experimentX)。"""
import re
from pathlib import Path

PLOT_DIR = Path("/Users/yubao/CODE/public/BBBD/analysis/code/plot")

SETUP = '''
# ---- 实验路径 (与 common.py / 07_lmm.R 对齐) ----
BBBD_EXPERIMENT <- Sys.getenv("BBBD_EXPERIMENT", unset = "2")
EXP_DIR <- switch(BBBD_EXPERIMENT,
                  "2" = "experiment2",
                  "3" = "experiment3",
                  "4" = "experiment4",
                  "5" = "experiment5",
                  "experiment2")
SCRIPT_DIR <- normalizePath(dirname(sub("--file=", "", grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE)[1])))
ANALYSIS_DIR <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
DATA_DIR <- file.path(ANALYSIS_DIR, "data", EXP_DIR)
FIG_DIR  <- file.path(ANALYSIS_DIR, "figures", EXP_DIR)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
'''

for p in sorted(PLOT_DIR.glob("*.R")):
    txt = p.read_text(encoding="utf-8")
    if "BBBD_EXPERIMENT" in txt and "DATA_DIR" in txt:
        print(f"跳过(已改造): {p.name}")
        continue
    # 注入 setup: 插到第一个 suppressMessages({...}) 块之后
    m = re.search(r"suppressMessages\(\{.*?\}\)", txt, flags=re.DOTALL)
    if m:
        txt = txt[:m.end()] + SETUP + txt[m.end():]
    else:
        # 没有 suppressMessages, 插到 #!/usr/bin 之后
        txt = txt.replace("#!/usr/bin/env Rscript\n", "#!/usr/bin/env Rscript\n" + SETUP, 1)

    # 替换路径: "data/xxx.csv" -> file.path(DATA_DIR, "xxx.csv")
    txt = re.sub(r'"data/([^"]+)"', r'file.path(DATA_DIR, "\1")', txt)
    # "figures/xxx.pdf" -> file.path(FIG_DIR, "xxx.pdf")
    txt = re.sub(r'"figures/([^"]+)"', r'file.path(FIG_DIR, "\1")', txt)
    # dir.create("figures") -> dir.create(FIG_DIR) (redundant but harmless)
    txt = txt.replace('dir.create("figures", showWarnings = FALSE)',
                      'dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)')

    p.write_text(txt, encoding="utf-8")
    print(f"已改造: {p.name}")

print("完成。")
