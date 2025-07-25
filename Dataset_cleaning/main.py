from src.utils.metrics import MetricsEvaluator

if __name__ == "__main__":
    gt_dir = "/home/sahithi_kukkala/indicDLP/data/indic_train_labels/train_ground_truths"
    pred_dir = "/home/sahithi_kukkala/indicDLP/data/indic_train_labels/train_result_labels2"
    image_dir = "/home/sahithi_kukkala/indicDLP/data/indic_data/images/train"
    save_path = "results/document_map_scores.csv"

    evaluator = MetricsEvaluator(gt_dir, pred_dir, image_dir)
    results = evaluator.evaluate_all(save_path=save_path)

    print(f"Saved results for {len(results)} documents to {save_path}")