"""
Script 04: Comprehensive evaluation on test split across multiple protocols (Section 9).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from qdml.data.dataset import DefectCropDataset
from qdml.models.backbone import QDMLNet
from qdml.quantum.distances import pairwise_quantum_similarity_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from qdml.eval.pipeline import (
    compute_retrieval_metrics,
    evaluate_knn,
    evaluate_few_shot,
    evaluate_embedding_geometry,
)


def extract_split_features(model: QDMLNet, dataset: DefectCropDataset, device: torch.device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model.eval()

    classical_feats = []
    quantum_states = []
    quantum_zs = []
    labels = []
    sources = []
    crop_ids = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            out = model(imgs)

            classical_feats.append(out["x"].cpu().numpy())
            if out["psi"] is not None:
                quantum_states.append(out["psi"])
                quantum_zs.append(out["z"].cpu().numpy())

            labels.append(batch["label"].numpy())
            sources.append(np.array(batch["source_image"]))
            crop_ids.extend(batch["crop_id"])

    res = {
        "x": np.concatenate(classical_feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "sources": np.concatenate(sources, axis=0),
        "crop_ids": crop_ids,
        "psi": torch.cat(quantum_states, dim=0) if quantum_states else None,
        "z": np.concatenate(quantum_zs, axis=0) if quantum_zs else None,
    }
    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluate QDML Run")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    parser.add_argument("--data", type=str, default="data/processed")
    args = parser.parse_args()

    run_dir = Path(args.run)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    cfg = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating checkpoint from {run_dir} on {device}...")

    # Load model
    model = QDMLNet(
        embed_dim=cfg.get("embed_dim", 32),
        n_qubits=cfg.get("n_qubits", 6),
        n_layers=cfg.get("n_layers", 2),
        pretrained=False,
        slice_mode=cfg.get("slice_mode", "learned"),
        use_classical_head=cfg.get("use_classical_head", True),
    ).to(device)

    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    manifest_path = Path(args.data) / "manifest.csv"
    train_ds = DefectCropDataset(manifest_path, split="train", is_training=False)
    test_ds = DefectCropDataset(manifest_path, split="test", is_training=False)

    print("Extracting features for train gallery and test queries...")
    train_data = extract_split_features(model, train_ds, device)
    test_data = extract_split_features(model, test_ds, device)

    # 1. Classical Space Retrieval
    # Headline: cross_split (query = test, gallery = train)
    sim_cross_c = np.dot(test_data["x"], train_data["x"].T)
    retrieval_cross_c = compute_retrieval_metrics(
        similarity_matrix=sim_cross_c,
        query_labels=test_data["labels"],
        gallery_labels=train_data["labels"],
    )

    # Leave-one-out (query = test, gallery = test with same-source exclusions)
    sim_loo_c = np.dot(test_data["x"], test_data["x"].T)
    np.fill_diagonal(sim_loo_c, -np.inf)
    retrieval_loo_c = compute_retrieval_metrics(
        similarity_matrix=sim_loo_c,
        query_labels=test_data["labels"],
        gallery_labels=test_data["labels"],
        query_sources=test_data["sources"],
        gallery_sources=test_data["sources"],
        exclude_same_source=True,
    )

    results = {
        "classical": {
            "cross_split": retrieval_cross_c,
            "leave_one_out": retrieval_loo_c,
            "knn": evaluate_knn(train_data["x"], train_data["labels"], test_data["x"], test_data["labels"]),
            "geometry": evaluate_embedding_geometry(test_data["x"], test_data["labels"]),
        }
    }

    # 2. Quantum Space Retrieval (if quantum model)
    if test_data["psi"] is not None and train_data["psi"] is not None:
        print("Evaluating Quantum Hilbert State Fidelity metrics...")
        # Compute exact state fidelities |<psi_q | psi_g>|^2
        sim_cross_q = pairwise_quantum_similarity_matrix("fidelity", test_data["psi"], train_data["psi"]).cpu().numpy()
        retrieval_cross_q = compute_retrieval_metrics(
            similarity_matrix=sim_cross_q,
            query_labels=test_data["labels"],
            gallery_labels=train_data["labels"],
        )

        sim_loo_q = pairwise_quantum_similarity_matrix("fidelity", test_data["psi"]).cpu().numpy()
        np.fill_diagonal(sim_loo_q, -np.inf)
        retrieval_loo_q = compute_retrieval_metrics(
            similarity_matrix=sim_loo_q,
            query_labels=test_data["labels"],
            gallery_labels=test_data["labels"],
            query_sources=test_data["sources"],
            gallery_sources=test_data["sources"],
            exclude_same_source=True,
        )

        results["quantum"] = {
            "cross_split": retrieval_cross_q,
            "leave_one_out": retrieval_loo_q,
        }

    # 3. Few-shot evaluation
    results["few_shot"] = evaluate_few_shot(test_data["x"], test_data["labels"])

    # 4. Classification Precision/Recall/F1 via kNN@1
    CLASS_NAMES = [
        "blurred_meter_reading",
        "damaged_cover_plate",
        "damaged_meter_housing",
        "fractured_silica_gel_body",
        "silica_gel_discoloration",
    ]
    knn1 = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn1.fit(train_data["x"], train_data["labels"])
    preds_knn1 = knn1.predict(test_data["x"])
    clf_report_str = classification_report(
        test_data["labels"], preds_knn1, target_names=CLASS_NAMES, zero_division=0
    )
    clf_report_dict = classification_report(
        test_data["labels"], preds_knn1, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    results["classification_report_knn1"] = clf_report_dict

    with open(eval_dir / "evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("Evaluation Results Summary (cross_split headline):")
    print(f"Classical R@1: {retrieval_cross_c['Recall@1']:.4f} | mAP: {retrieval_cross_c['mAP']:.4f}")
    if "quantum" in results:
        q_res = results["quantum"]["cross_split"]
        print(f"Quantum   R@1: {q_res['Recall@1']:.4f} | mAP: {q_res['mAP']:.4f}")
    print("\n--- kNN@1 Classification Report (test split) ---")
    print(clf_report_str)
    print("=" * 50)


if __name__ == "__main__":
    main()
