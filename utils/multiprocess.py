import multiprocessing as mp

def multiprocess_wrapper(function, items, threads):

    ###########################################
    # Function: Parallelise function.         #
    #                                         #
    # Inputs: Function to parallelise         #
    # (def score),                            #
    # list of tuples as function input,       #
    # number of threads to parallelise across #
    #                                         #
    # Output: List of returned results        #
    ###########################################

    processes = min(threads, mp.cpu_count())
    with mp.Pool(processes) as p:
        r = list(p.imap(function, items))
        p.close()
        p.join()

    return r
