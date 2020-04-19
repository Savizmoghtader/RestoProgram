import timeit
import os

from Engine import Engine
from PP_Engine import PpEngine

if __name__ == '__main__':
    print('***** start the simulation *****')
    start = timeit.default_timer()
    resultDir = './Testresults_4PSO_12/'
    model = Engine(resultDir, bTest=False, bRestConstraint=False)
    model.run()
    # ppmodel = PpEngine(resultDir, bTest=False)
    # ppmodel.run('3')
    # os.system('cd' + resultDir + 'tex/ && pdflatex fig-3.tex')
    stop = timeit.default_timer()
    print('time: ' + str(stop - start))
    print('****** simulation finished ******')