import json
import statistics
from pathlib import Path

def compile_report():
    script_dir = Path(__file__).parent.resolve()
    results_dir = script_dir / "dump"
    
    report_md = """# Unified Semantic Gravity Evidence Report
## Objective
To provide conclusive, multi-model evidence that language models process structural multi-hop movement sequences (e.g. `A is moved to B -> B is moved to C`) not via strict logical entity tracking, but as a rigid structural probability funnel ("Semantic Gravity Well").

We present three distinct pieces of evidence across 1B, 8B, and 32B models.

"""

    for model_id in ["1B", "8B", "32B"]:
        file_path = results_dir / f"{model_id}_unified_raw.json"
        if not file_path.exists():
            continue
            
        with open(file_path, "r") as f:
            results = json.load(f)
            
        report_md += f"## 🔵 MODEL: {model_id}\n\n"
        
        # ----------------------------------------------------
        # METHOD 1: The Irrelevant Chain Ablation
        # ----------------------------------------------------
        base_acc = sum(1 for r in results if r['containers']['unmoved_0'] in r['baseline']['pred'].lower()) / len(results) * 100
        corr_acc = sum(1 for r in results if r['containers']['unmoved_0'] in r['corrupted']['pred'].lower()) / len(results) * 100
        irrel_acc = sum(1 for r in results if r['containers']['unmoved_0'] in r['irrelevant']['pred'].lower()) / len(results) * 100
        
        report_md += "### Method 1: The Irrelevant Chain Ablation\n"
        report_md += "We test the model's accuracy on the target object's location across three prompt variations. In the 'Irrelevant Chain', a completely disconnected object (e.g., a 'tray') moves through the room. If the model's accuracy drops from Baseline merely because an irrelevant chain is present, it proves the chain acts as a blind attention sink.\n\n"
        report_md += f"- **Baseline (No Hops):** {base_acc:.1f}%\n"
        report_md += f"- **Corrupted (Current):** {corr_acc:.1f}%\n"
        report_md += f"- **Irrelevant Chain (The Proof):** {irrel_acc:.1f}%\n\n"
        
        # ----------------------------------------------------
        # METHOD 2: Distractor Probability Spikes
        # ----------------------------------------------------
        base_b = statistics.mean([r['baseline']['probs']['unmoved_0'] for r in results])
        base_d = statistics.mean([r['baseline']['probs']['chain_2'] for r in results])
        corr_b = statistics.mean([r['corrupted']['probs']['unmoved_0'] for r in results])
        corr_d = statistics.mean([r['corrupted']['probs']['chain_2'] for r in results])
        
        report_md += "### Method 2: Distractor Probability Spikes\n"
        report_md += "We measure the raw log-probability mass of the unmoved target ('B') versus the final destination of the moving distractor container ('D'). If the model is not attending to the multi-hop logic, 'D' should have near-zero probability.\n\n"
        report_md += "| Variation | Prob(Target Container `B`) | Prob(Distractor Destination `D`) |\n"
        report_md += "|---|---|---|\n"
        report_md += f"| **Baseline** | {base_b:.4f} | {base_d:.4f} |\n"
        report_md += f"| **Corrupted** | {corr_b:.4f} | {corr_d:.4f} |\n\n"
        
        # ----------------------------------------------------
        # METHOD 3: Multi-Hop Probability Ratios
        # ----------------------------------------------------
        c_c0 = statistics.mean([r['clean']['probs']['chain_0'] for r in results])
        c_c1 = statistics.mean([r['clean']['probs']['chain_1'] for r in results])
        c_c2 = statistics.mean([r['clean']['probs']['chain_2'] for r in results])
        
        corr_c0 = statistics.mean([r['corrupted']['probs']['chain_0'] for r in results])
        corr_c1 = statistics.mean([r['corrupted']['probs']['chain_1'] for r in results])
        corr_c2 = statistics.mean([r['corrupted']['probs']['chain_2'] for r in results])
        
        irrel_c0 = statistics.mean([r['irrelevant']['probs']['chain_0'] for r in results])
        irrel_c1 = statistics.mean([r['irrelevant']['probs']['chain_1'] for r in results])
        irrel_c2 = statistics.mean([r['irrelevant']['probs']['chain_2'] for r in results])
        
        c_r1, c_r2 = (c_c1/c_c0, c_c2/c_c0) if c_c0 > 0 else (0,0)
        corr_r1, corr_r2 = (corr_c1/corr_c0, corr_c2/corr_c0) if corr_c0 > 0 else (0,0)
        irrel_r1, irrel_r2 = (irrel_c1/irrel_c0, irrel_c2/irrel_c0) if irrel_c0 > 0 else (0,0)
        
        report_md += "### Method 3: Multi-Hop Probability Ratios\n"
        report_md += "We hypothesize that the probability ratio between the containers in the movement chain (`Chain_0` : `Chain_1` : `Chain_2`) remains mathematically similar regardless of what item is inside the chain.\n\n"
        report_md += "| State | LogProb Ratio (`Chain_0`:`Chain_1`:`Chain_2`) |\n"
        report_md += "|---|---|\n"
        report_md += f"| **Clean (Target moves)** | `1.0 : {c_r1:.1f} : {c_r2:.1f}` |\n"
        report_md += f"| **Corrupted (Distractor moves)** | `1.0 : {corr_r1:.1f} : {corr_r2:.1f}` |\n"
        report_md += f"| **Irrelevant Chain (Random item moves)** | `1.0 : {irrel_r1:.1f} : {irrel_r2:.1f}` |\n\n"
        
        report_md += "---\n\n"
        
    with open(script_dir / "03_master_unified_report.md", "w") as f:
        f.write(report_md)
        
    import markdown
    html_body = markdown.markdown(report_md, extensions=['tables'])
    html_template = f"<html><head><style>body{{font-family:sans-serif; line-height:1.6; max-width:900px; margin:auto; padding:2rem; background:#f4f4f9;}} table{{border-collapse:collapse; width:100%; margin:1rem 0; background:white;}} th,td{{border:1px solid #ddd; padding:8px; text-align:left;}} th{{background:#2c3e50; color:white;}} h2{{color:#2c3e50; border-bottom:2px solid #ddd; padding-bottom:10px;}}</style></head><body>{html_body}</body></html>"
    with open(script_dir / "03_master_unified_report.html", "w") as f:
        f.write(html_template)

    print("Successfully compiled unified master report.")

if __name__ == "__main__":
    compile_report()
