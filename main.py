import timeit



if __name__ == '__main__':
    print('***** start the simulation *****')
    start = timeit.default_timer()
    model = Engine('./results/')
    model.run()
    stop = timeit.default_timer()
    print('time: ' + str(stop - start))
    print('****** simulation finished ******')