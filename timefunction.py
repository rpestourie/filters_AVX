import timeit
import numpy as np

def time_update(function, truncate, imsize, picture, input_im, sigma_r, sigma_s, lw, num_thread=None):

    #cython parameters
    imsize0 = imsize[0]
    imsize1 = imsize[1]
    output = picture*0.
    output5 =  np.array(output, np.float32)
    input_im5 = np.array(input_im, np.float32)

    if num_thread is None:
        times = timeit.repeat(lambda: function(sigma_s,
                                                        sigma_r,
                                                        input_im5,
                                                        imsize0,
                                                        imsize1,
                                                        output5,
                                                        lw),
                            number=3, repeat=5)
    else:
        times = timeit.repeat(lambda: function(sigma_s,
                                                    sigma_r,
                                                    input_im5,
                                                    imsize0,
                                                    imsize1,
                                                    output5,
                                                    lw,
                                                    num_thread),
                            number=3, repeat=5)
    print("{}: {}s".format(str(function) , min(times)))

    return min(times)
