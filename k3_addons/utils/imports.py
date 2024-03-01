# create a variable and check if pysoti is available and return a constant PYSOTI_AVAILABLE

try:
    import pystoi

    PYSOTI_AVAILABLE = True
except ImportError:
    PYSOTI_AVAILABLE = False

try:
    import pesq

    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

try:
    import multiprocessing

    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
