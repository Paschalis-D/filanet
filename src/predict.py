"""Write a competition submission from trained checkpoints.

The test photographs must go through the same preprocessing as training first:
    python img_prepro_v2.py <test_images> processed/test
Any photograph that preprocessing rejected is missing from processed/test and so
from the submission, which costs every one of its filaments as a miss - check
its failure list before submitting.

Then, with the operating point and merge gap sweep.py chose:
    python predict.py --checkpoint runs/x/best.pt --tta --point 0.95,300,0.5 --merge-gap 20 \
        --images processed/test --out submission.csv
"""
import argparse
from pathlib import Path

import torch

from dataset import FilamentDataset
from inference import load_model, predict_probability
from submission import point_name, probability_to_instances, rows_from_instances, write_submission


def parse_point(text: str) -> tuple:
    """'0.95,600' or '0.95,300,0.5' -> an operating point, as point_name prints them."""
    values = text.split(",")
    if len(values) not in (2, 3):
        raise argparse.ArgumentTypeError("expected threshold,min_area or high,min_area,low")
    high, min_area = float(values[0]), int(values[1])
    return (high, min_area) if len(values) == 2 else (high, min_area, float(values[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True,
                        help="one or more checkpoints; several are averaged")
    parser.add_argument("--images", type=Path, required=True, help="preprocessed test images")
    parser.add_argument("--point", type=parse_point, required=True,
                        help="threshold,min_area or high,min_area,low")
    parser.add_argument("--tta", action="store_true", help="average all eight orientations")
    parser.add_argument("--merge-gap", type=int, default=0,
                        help="join predicted pieces at most this many px apart into one "
                             "filament; sweep.py reports the best value")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_model(p, device) for p in args.checkpoint]
    data = FilamentDataset(args.images, is_train=False)

    rows = []
    for index in range(len(data)):
        image, file_name = data[index]
        with torch.no_grad():
            probability = predict_probability(models, image.unsqueeze(0).to(device), args.tta)
        instances = probability_to_instances(probability.cpu().numpy(), args.point, args.merge_gap)
        rows.extend(rows_from_instances(Path(file_name).stem, instances))
        print(f"  {index + 1}/{len(data)}  {len(rows)} filaments so far", end="\r", flush=True)
    print()

    write_submission(rows, args.out)
    print(f"{len(rows)} filaments over {len(data)} photographs at {point_name(args.point)}, "
          f"merge gap {args.merge_gap} px -> {args.out}")


if __name__ == "__main__":
    main()
