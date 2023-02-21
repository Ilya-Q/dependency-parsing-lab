from models.perceptron import StructuredPerceptron
from decoders.eisner import eisner
from conll import read_conll
from features import TemplateFeatures
from pathlib import Path
import multiprocessing
import pickle
import lzma
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=read_conll)
parser.add_argument("--num_epochs", "-n", type=int,default=1)
parser.add_argument("--model_file", "-m", type=Path)

if __name__ == '__main__':
    args = parser.parse_args()

    perceptron = StructuredPerceptron(eisner)

    start = time.monotonic()
    print("Extracting features...", end='')
    with multiprocessing.Pool() as p:
        feats = p.map(TemplateFeatures, args.data)
    elapsed = time.monotonic() - start
    print(f" done in {elapsed:.3} seconds!")

    perceptron.fit(feats, epochs=args.num_epochs)

    start = time.monotonic()
    print(f"Saving model to {args.model_file}...", end='')
    with lzma.open(args.model_file, 'w') as f:
        pickle.dump(perceptron, f)
    elapsed = time.monotonic() - start
    print(f" done in {elapsed:.3} seconds!")