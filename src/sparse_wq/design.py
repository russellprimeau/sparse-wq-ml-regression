"""Two predictor sets with the same change-from-prior response."""
RUNS = {
    'noprofiler_differential': ('no profiler', 'difference', 'offered'),
    'profiler_differential': ('profiler', 'difference', 'offered'),
}
ARM_KEYS = (('noprofiler_differential','shared'),('profiler_differential','off'))
RUN_ORDER = tuple(run for run,_ in ARM_KEYS)
RUN_LABEL = {'noprofiler_differential':'No profiler; change from prior',
             'profiler_differential':'Profiler; change from prior'}
def offers_profiler(run): return RUNS[run][0] == 'profiler'
def is_absolute(run): return False
