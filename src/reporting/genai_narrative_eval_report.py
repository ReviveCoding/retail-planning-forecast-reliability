from __future__ import annotations
import pandas as pd
from src.utils import write_md


def write_genai_eval_report(eval_df: pd.DataFrame, narrative: str, path='reports/genai_narrative_eval_report.md'):
    text = "# GenAI Planning Narrative Evaluation Report\n\n"
    text += "## Clean narrative\n\n" + narrative + "\n\n"
    text += "## Evaluation metrics and robustness cases\n\n" + eval_df.to_markdown(index=False)
    text += "\n\n## Interpretation\n\nThe clean template should pass evidence-grounding checks, while corrupted narratives are expected to trigger lower numeric consistency, driver faithfulness, caveat coverage, or human-review routing. This makes the narrative layer an evaluation harness rather than a cosmetic summary generator.\n"
    write_md(text, path)
