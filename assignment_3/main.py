from .tasks.clean_speech_classifier import run_1
from .tasks.noisy_speech_classification import run_2
from .tasks.reverb_der_classifier import run_3

# run_1()
# run_2()
# run_3()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support() 
    run_1()
    run_2()
    run_3()
