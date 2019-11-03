import timeit
from Engine import Engine


if __name__ == '__main__':
    print('***** start the simulation *****')
    start = timeit.default_timer()
    model = Engine('./results/', bTest=True)
    model.run()
    stop = timeit.default_timer()
    print('time: ' + str(stop - start))
    print('****** simulation finished ******')