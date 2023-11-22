import json
import os

import hydra
import pytorch_lightning as pl
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

SPLIT = False

def split_predictions(predictions):
    q_dict = {
        "object": [],
        "color": [],
        "nature": [],
        "place": [],
        "number": [],
        "other": []
    }
    for pred in predictions:
        if "What is" in pred["question"] and ("color " not in pred["question"] or "What is on the left of the light wood color table?" == pred["question"]):
            q_dict["object"].append(pred)
        elif "What color" in pred["question"] or "What is the color" in pred["question"]:
            q_dict["color"].append(pred)
        elif "What type" in pred["question"] or "What shape" in pred["question"] or "What kind" in pred["question"]:
            q_dict["nature"].append(pred)
        elif "Where" in pred["question"]:
            q_dict["place"].append(pred)
        elif "How many" in pred["question"]:
            q_dict["number"].append(pred)
        else:
            q_dict["other"].append(pred)
            
    return q_dict

def score(predictions):
    tokenizer = PTBTokenizer()

    print("==> formatting data ...")
    res = {}
    gts = {}

    for pred in predictions:
        question_id = pred["question_id"]
        res[question_id] = [{"caption": pred["answer_top10"][0]}]
        gts[question_id] = [{"caption": answer} for answer in pred["gt_answers"]]

    res = tokenizer.tokenize(res)
    gts = tokenizer.tokenize(gts)

    # accuracies
    # EM@1
    total = 0
    count = 0
    for pred in predictions:
        if pred["answer_top10"][0] in pred["gt_answers"]:
            count += 1
        total += 1
    top_one_accuracy = count / total
    print(f"EM@1: {round(top_one_accuracy * 100, 2)}")
    # EM@10
    total = 0
    count = 0
    for pred in predictions:
        a = set(pred["answer_top10"])
        b = set(pred["gt_answers"])
        intersect = a & b
        if len(intersect) > 0:
            count += 1
        total += 1
    top_ten_accuracy = count / total
    print(f"EM@10: {round(top_ten_accuracy * 100, 2)}")

    # captioning metrics
    scores_str = ""
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "Spice"),
    ]
    scores_results = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, _, m in zip(score, scores, method):
                scores_results[m] = round(sc * 100, 2)
                value = round(sc * 100, 2)
                scores_str += "& " + str(value) + " "
                print(f"{m}: {value}")
        else:
            scores_results[method] = round(score * 100, 2)
            value = round(score * 100, 2)
            scores_str += "& " + str(value) + " "
            print(f"{method}: {value}")
    print(scores_str)
    
    # object localization
    print("==> scoring ...")
    ious = [prediction["iou"] for prediction in predictions]
    acc_25 = len([iou for iou in ious if iou >= 0.25]) / len(ious)
    acc_5 = len([iou for iou in ious if iou >= 0.5]) / len(ious)
    print(f"Acc@0.25: {round(acc_25 * 100, 2)}")
    print(f"Acc@0.5: {round(acc_5 * 100, 2)}")
    


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    #########

    ########

    # fix the seed
    pl.seed_everything(cfg.global_train_seed, workers=True)

    predictions_path = os.path.join(
        cfg.exp_output_root_path,
        cfg.model.experiment_name,
        "inference",
        "predictions.json",
    )

    if not os.path.exists(predictions_path):
        print(
            f"No inference predictions exists for experiment {cfg.model.experiment_name}."
        )
        return

    predictions = json.load(open(predictions_path))
    
    if SPLIT:
        q_dict = split_predictions(predictions)
        
        for k,v in q_dict.items():
            print(f"=============================={k} ({len(v)})==============================")
            score(v)
    
    print(f"==============================Total ({len(predictions)})==============================")
    score(predictions)


if __name__ == "__main__":
    main()
