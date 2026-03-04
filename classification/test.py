"""
Standalone classification inference.
Loads from ckpt_dir (e.g. Inference/ckpt/classification), saves to output_dir.
"""
import os
import sys
import argparse
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from dataset import SingleImageDataset, MultiImageDataset
from model import ClassificationModel, TIMM_AVAILABLE
from utils import load_single_data, load_multi_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
    return all_preds, all_labels, all_paths


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Classification inference (standalone)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Directory with fold_0, fold_1, ... (args.json + checkpoints/best.pth). Default: Inference/ckpt/classification')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save predictions.json, metrics.json, confusion_matrix.png')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Dataset root (valA+valB or valA+predB). 생략 시 --data_root_A, --data_root_B 사용')
    parser.add_argument('--data_root_A', type=str, default=None, help='A 이미지 폴더 (예: valA 또는 원본 경로)')
    parser.add_argument('--data_root_B', type=str, default=None, help='B 이미지 폴더 (예: stain2stain 결과 fake_B 또는 valB)')
    parser.add_argument('--mode', type=str, choices=['A', 'B', 'AB'], default='AB')
    parser.add_argument('--is_pred', action='store_true', help='Use valA+predB (data_root 사용 시에만)')
    parser.add_argument('--fold', type=str, default='all', help='0-4 or all')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    # Optional: used when args.json is missing (must match training config)
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', help='Backbone name (used if args.json not found)')
    parser.add_argument('--img_size', type=int, default=1024, help='Input size (used if args.json not found)')
    parser.add_argument('--ab_fusion_mode', type=str, default='concat', help='AB fusion mode (used if args.json not found)')
    parser.add_argument('--ab_weight_A', type=float, default=1.0, help='Weight for A in AB mode (used if args.json not found)')
    parser.add_argument('--ab_weight_B', type=float, default=0.1, help='Weight for B in AB mode (used if args.json not found)')
    args = parser.parse_args()

    # A/B 경로: data_root 하나 또는 data_root_A + data_root_B 둘 다 지정
    if args.mode == 'AB':
        if (args.data_root_A is not None and args.data_root_B is not None):
            args.use_separate_AB = True
        elif args.data_root:
            args.use_separate_AB = False
        else:
            raise ValueError('AB 모드: --data_root 를 쓰거나 --data_root_A 와 --data_root_B 를 둘 다 지정하세요.')
    elif args.mode == 'A':
        args.use_separate_AB = (args.data_root_A is not None)
        if not args.use_separate_AB and not args.data_root:
            raise ValueError('A 모드: --data_root 또는 --data_root_A 를 지정하세요.')
    else:  # B
        args.use_separate_AB = (args.data_root_B is not None)
        if not args.use_separate_AB and not args.data_root:
            raise ValueError('B 모드: --data_root 또는 --data_root_B 를 지정하세요.')

    # Default ckpt_dir: Inference/ckpt/classification (parent of classification/ is Inference when run from Inference/)
    if args.ckpt_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.ckpt_dir = os.path.join(base, 'ckpt', 'classification')
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'classification_out')
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.fold.lower() == 'all':
        folds_to_use = list(range(5))
        use_ensemble = True
    else:
        folds_to_use = [int(args.fold)]
        use_ensemble = False

    # Load training args from first fold (args.json optional; fallback to CLI)
    fold_name = f"fold_{folds_to_use[0]}"
    args_path = os.path.join(args.ckpt_dir, fold_name, 'args.json')
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            training_args = json.load(f)
        backbone = training_args.get('backbone', 'efficientnet_b0')
        img_size = int(training_args.get('img_size', 1024))
        ab_fusion_mode = training_args.get('ab_fusion_mode', 'concat')
        ab_weight_A = float(training_args.get('ab_weight_A', 1.0))
        ab_weight_B = float(training_args.get('ab_weight_B', 0.1))
    else:
        backbone = args.backbone
        img_size = args.img_size
        ab_fusion_mode = args.ab_fusion_mode
        ab_weight_A = args.ab_weight_A
        ab_weight_B = args.ab_weight_B

    models = []
    for fold_idx in folds_to_use:
        fold_name = f"fold_{fold_idx}"
        best_path = os.path.join(args.ckpt_dir, fold_name, 'checkpoints', 'best.pth')
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"Checkpoint not found: {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        model = ClassificationModel(
            num_classes=4,
            input_mode=args.mode,
            backbone=backbone,
            is_pretrained=False,
            ab_fusion_mode=ab_fusion_mode if args.mode == 'AB' else 'concat',
            ab_weight_A=ab_weight_A if args.mode == 'AB' else 1.0,
            ab_weight_B=ab_weight_B if args.mode == 'AB' else 1.0
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        model.eval()
        models.append(model)

    # Load data
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    if args.mode in ('A', 'B'):
        if args.use_separate_AB:
            test_dir = args.data_root_A if args.mode == 'A' else args.data_root_B
        else:
            if args.is_pred and args.mode == 'B':
                test_dir = os.path.join(args.data_root, 'predB')
            else:
                test_dir = os.path.join(args.data_root, 'val' + args.mode)
        image_paths, labels = load_single_data(test_dir)
        test_dataset = SingleImageDataset(image_paths, labels, transform=transform, return_paths=True)
    else:
        if args.use_separate_AB:
            testA_dir = os.path.abspath(args.data_root_A)
            testB_dir = os.path.abspath(args.data_root_B)
        else:
            if args.is_pred:
                testA_dir = os.path.join(args.data_root, 'valA')
                testB_dir = os.path.join(args.data_root, 'predB')
            else:
                testA_dir = os.path.join(args.data_root, 'valA')
                testB_dir = os.path.join(args.data_root, 'valB')
        image_pairs, labels = load_multi_data(testA_dir, testB_dir)
        test_dataset = MultiImageDataset(image_pairs, labels, transform=transform, return_paths=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if use_ensemble and len(models) > 1:
        all_preds_list = []
        all_labels = all_paths = None
        for model in models:
            preds, labels, paths = test(model, test_loader, device)
            all_preds_list.append(preds)
            if all_labels is None:
                all_labels, all_paths = labels, paths
        all_preds = [Counter(preds[i] for preds in all_preds_list).most_common(1)[0][0] for i in range(len(all_preds_list[0]))]
        all_preds = np.array(all_preds)
    else:
        all_preds, all_labels, all_paths = test(models[0], test_loader, device)

    accuracy = accuracy_score(all_labels, all_preds)
    class_names = ['0', '1+', '2+', '3+']
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    metrics = {'accuracy': accuracy, 'classification_report': report, 'num_samples': len(all_labels)}
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    predictions = []
    for pred, label, path in zip(all_preds, all_labels, all_paths):
        if args.mode == 'AB':
            predictions.append({'pathA': path[0], 'pathB': path[1], 'true_label': int(label), 'predicted_label': int(pred), 'correct': bool(pred == label)})
        else:
            predictions.append({'path': path, 'true_label': int(label), 'predicted_label': int(pred), 'correct': bool(pred == label)})
    with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
        json.dump(predictions, f, indent=2)

    plot_confusion_matrix(all_labels, all_preds, class_names, os.path.join(args.output_dir, 'confusion_matrix.png'))
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
