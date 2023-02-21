import multiprocessing

def decode_parallel(decoder, input):
    with multiprocessing.Pool() as p:
        return p.starmap(decoder, input)