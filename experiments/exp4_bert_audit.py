# experiments/exp5_cross_audit.py

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import torch
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isqed.real_world import HuggingFaceWrapper, MaskingIntervention
from isqed.geometry import DISCOSolver


def run_cross_audit_dual_mode():
    print("--- Running Exp 5: Full Ecosystem Dual-Mode Audit (DISCO Standard) ---")
    
    device = "cpu"
    print(f"Using device: {device}")
    
    # =========================================================================
    # 1. Define Models
    # =========================================================================
    model_ids = [
        "textattack/bert-base-uncased-SST-2", "textattack/distilbert-base-uncased-SST-2",
        "textattack/roberta-base-SST-2", "textattack/albert-base-v2-SST-2",
        "textattack/xlnet-base-cased-SST-2"
    ]
    short_names = ["BERT", "DistilBERT", "RoBERTa", "ALBERT", "XLNet"]
    
    original_models = []
    print("Loading Core Ecosystem Models...")
    for pid in model_ids:
        try:
            m = HuggingFaceWrapper(pid, device)
            original_models.append(m)
        except Exception as e:
            print(f"  [FAIL] Skipping {pid}: {e}")
            
    n = len(original_models)
    if n < 3:
        print("Less than 3 models loaded. Aborting experiment.")
        return

    # =========================================================================
    # 3. Data and Dose Design (P_fit vs P_eval) - Using your structure
    # =========================================================================
    intervention = MaskingIntervention()

    print("Loading SST-2 validation data...")
    try:
        dataset = load_dataset("glue", "sst2", split="validation")
        all_sentences = dataset["sentence"]
        max_samples = 200
        all_sentences = all_sentences[:max_samples]
    except:
        all_sentences = ["This sentence is a test sentence"] * 200

    rng = np.random.RandomState(0)
    all_sentences = np.array(all_sentences)
    rng.shuffle(all_sentences)
    n_total = len(all_sentences)
    n_fit = n_total // 2
    fit_texts = all_sentences[:n_fit].tolist()
    eval_texts = all_sentences[n_fit:].tolist()

    print(f"Total sentences: {n_total}, fit: {len(fit_texts)}, eval: {len(eval_texts)}")

    doses_fit = np.linspace(0.0, 0.3, 4)
    doses_eval = np.linspace(0.4, 0.88, 5)

    print(f"P_fit doses (low): {doses_fit}")
    print(f"P_eval doses (high): {doses_eval}")

    # =========================================================================
    # 4. Main Loop: DISCO-style Dual Audit
    # =========================================================================
    
    # Store residuals for the final heatmap visualization (rows = samples*doses, cols = targets*modes)
    final_residual_data = pd.DataFrame()
    
    # Generate the sample index column based on P_eval length (required for the final heatmap)
    n_eval_points = len(eval_texts) * len(doses_eval)
    eval_index = []
    for i in range(len(eval_texts)):
        for j in range(len(doses_eval)):
            eval_index.append(f"Sample {i+1} $\\theta={doses_eval[j]:.2f}$")
            
    final_residual_data['Eval_Point'] = eval_index


    for i in range(n):
        target_model = original_models[i]
        target_name = short_names[i]
        
        # --- Define Peer Sets for this Target ---
        
        # Peer Set 1 (Rest of Ecosystem, size N-1)
        peers_rest = [original_models[j] for j in range(n) if i != j]
        
        # Peer Set 2 (Full Ecosystem, size N)
        peers_full = original_models
        
        print(f"\n>>> Auditing target: {target_name}")

        
        # --- 4.1 FIT PHASE (Learn w_hat) ---
        y_t_fit_list = []
        Y_p_fit_rest_list = []
        Y_p_fit_full_list = []

        print("  [Phase] Fitting convex baseline on low-dose interventions (P_fit)...")
        
        # Collect data for fitting the two weight vectors (w_rest and w_full)
        for text in tqdm(fit_texts, desc="    Fit texts"):
            for theta in doses_fit:
                seed = abs(hash((text, float(theta)))) % (2**32)
                perturbed_text = intervention.apply(text, theta, seed=seed)
                
                y_t = target_model._forward(perturbed_text)
                
                # Query peers for both modes
                y_ps_rest = [p._forward(perturbed_text) for p in peers_rest]
                y_ps_full = [p._forward(perturbed_text) for p in peers_full]
                
                y_t_fit_list.append(y_t)
                Y_p_fit_rest_list.append(y_ps_rest)
                Y_p_fit_full_list.append(y_ps_full)


        y_t_fit_vec = np.array(y_t_fit_list, dtype=float) 
        if y_t_fit_vec.ndim == 1: y_t_fit_vec = y_t_fit_vec.reshape(-1, 1) 
        
        # Solve W_hat_rest (Mode 1: Uniqueness)
        Y_p_fit_rest_mat = np.array(Y_p_fit_rest_list, dtype=float) 
        _, w_hat_rest = DISCOSolver.solve_weights_and_distance(y_t_fit_vec, Y_p_fit_rest_mat)
        
        # Solve W_hat_full (Mode 2: Redundancy)
        Y_p_fit_full_mat = np.array(Y_p_fit_full_list, dtype=float) 
        _, w_hat_full = DISCOSolver.solve_weights_and_distance(y_t_fit_vec, Y_p_fit_full_mat)

        print(f"  [Fit] Learned w_rest for {target_name}: {w_hat_rest.flatten()[:2]}...")
        print(f"  [Fit] Learned w_full for {target_name}: {w_hat_full.flatten()[:2]}...")

        # --- 4.2 EVAL PHASE (Compute Residuals) ---
        
        r_uniqueness_list = []
        r_redundancy_list = []

        print("  [Phase] Evaluating PIER on high-dose interventions (P_eval)...")
        
        for text in tqdm(eval_texts, desc="    Eval texts"):
            for theta in doses_eval:
                seed = abs(hash((text, float(theta)))) % (2**32)
                perturbed_text = intervention.apply(text, theta, seed=seed)

                # Query Target (y_t is the same for both modes)
                y_t = target_model._forward(perturbed_text)

                # Query Peers (y_ps_rest and y_ps_full)
                y_ps_rest = np.array([p._forward(perturbed_text) for p in peers_rest], dtype=float)
                y_ps_full = np.array([p._forward(perturbed_text) for p in peers_full], dtype=float)
                
                # Mode 1: Residual R_rest using FIXED w_hat_rest
                y_mix_rest = np.dot(w_hat_rest.flatten(), y_ps_rest)
                r_uniqueness_list.append(float(np.abs(y_t - y_mix_rest)))

                # Mode 2: Residual R_full using FIXED w_hat_full
                y_mix_full = np.dot(w_hat_full.flatten(), y_ps_full)
                r_redundancy_list.append(float(np.abs(y_t - y_mix_full)))

        # --- Save Final Residual Vectors (Full N_samples*N_doses) ---
        final_residual_data[f'{target_name}_Uniqueness_R'] = r_uniqueness_list
        final_residual_data[f'{target_name}_Redundancy_R'] = r_redundancy_list


    # 5. Save
    output_dir = "results/tables"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "exp5_dual_mode_residuals.csv")
    final_residual_data.to_csv(out_path, index=False)
    print(f"\nSaved raw residuals data for {n_eval_points} samples to: {out_path}")

if __name__ == "__main__":
    run_cross_audit_dual_mode()