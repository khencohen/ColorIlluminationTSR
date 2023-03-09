__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '1.3.2023'

from TSRmathSimulatorUtils import *


def compare_x1_x3_x4_x5_x6_score(epochs, min_f = 0, max_f = 0, super_time_step = 0.):
    total_freq_vec = []
    total_l2_score_vec = []

    for epoch in range(epochs):
        freq = min_f + np.random.rand() * (max_f-min_f)
        f_fun = lambda t: freq * np.sin(freq * 2 * np.pi * t)

        print('epoch =', epoch)
        #### True Signal #####
        sigTrue = SignalClass(np.array(f_fun(super_time_vec[:-int(1 / super_time_step)])),
                              time_step=super_time_step / camera_fps, name='True Signal')

        #### N = 1 #####
        signal_vec_x1 = sample_function(f_fun, time_vec)

        #### N = 3 #####
        b_vec, g_vec, r_vec = get_bgr_vectors_from_N(3)
        signal_vec_x3 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        #### N = 4 #####
        b_vec, g_vec, r_vec = get_bgr_vectors_from_N(4)
        signal_vec_x4 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        #### N = 5 #####
        b_vec, g_vec, r_vec = get_bgr_vectors_from_N(5)
        signal_vec_x5 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        #### N = 6 #####
        b_vec, g_vec, r_vec = get_bgr_vectors_from_N(6)
        signal_vec_x6 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        # Temporal up-sampling:
        signal_vec_x1 = temporal_dense_vec(signal_vec_x1, int(1 / (1 * super_time_step)))
        signal_vec_x3 = temporal_dense_vec(signal_vec_x3, int(1 / (3 * super_time_step)))
        signal_vec_x4 = temporal_dense_vec(signal_vec_x4, int(1 / (4 * super_time_step)))
        signal_vec_x5 = temporal_dense_vec(signal_vec_x5, int(1 / (5 * super_time_step)))
        signal_vec_x6 = temporal_dense_vec(signal_vec_x6, int(1 / (6 * super_time_step)))

        sig1x = SignalClass(np.array(signal_vec_x1), time_step=super_time_step / camera_fps, name='x1')
        sig3x = SignalClass(np.array(signal_vec_x3), time_step=super_time_step / camera_fps, name='x3')
        sig4x = SignalClass(np.array(signal_vec_x4), time_step=super_time_step / camera_fps, name='x4')
        sig5x = SignalClass(np.array(signal_vec_x5), time_step=super_time_step / camera_fps, name='x5')
        sig6x = SignalClass(np.array(signal_vec_x6), time_step=super_time_step / camera_fps, name='x6')

        # Compare perforemence:
        l2_scores = [sigTrue.l2_distance(sig1x, int(0.5 / (super_time_step))), \
                     sigTrue.l2_distance(sig3x, int(0.5 / (3 * super_time_step))), \
                     sigTrue.l2_distance(sig4x, int(0.5 / (4 * super_time_step))), \
                     sigTrue.l2_distance(sig5x, int(0.5 / (5 * super_time_step))), \
                     sigTrue.l2_distance(sig6x, int(0.5 / (6 * super_time_step)))]

        total_freq_vec += [[freq] * 5]
        total_l2_score_vec += [l2_scores]

    plt.plot(np.array(total_freq_vec), np.array(total_l2_score_vec), 'o')
    plt.legend(['x1', 'x3', 'x4', 'x5', 'x6'])
    plt.title('L2 Score')
    plt.show()

    return

def compare_fixed_pattern_flicker_vec(epochs, min_f, max_f, N = 4, super_time_step = 0.):
    if N == 4:
        flicker_mat_vec = [ np.array([[1, 0, 0, 1],\
                                      [0, 1, 0, 0],\
                                      [0, 0, 1, 0]]),\
                            np.array([[1, 0, 0, 1], \
                                      [1, 0, 1, 0], \
                                      [0, 1, 0, 1]]),\
                            np.array([[1, 0, 0, 0], \
                                      [0, 1, 1, 0], \
                                      [0, 0, 0, 1]]),\
                            np.array([[1, 1, 0, 0], \
                                      [0, 1, 1, 0], \
                                      [0, 0, 1, 1]]),\
                            np.array([[0, 0, 1, 0], \
                                      [0, 1, 0, 0], \
                                      [1, 1, 1, 1]]) ]
    elif N == 5:
        flicker_mat_vec = [ np.array([[1, 0, 0, 0, 1],\
                                      [0, 1, 1, 0, 0],\
                                      [0, 0, 1, 1, 0]]),\
                            np.array([[1, 0, 0, 1, 0], \
                                      [1, 0, 1, 0, 1], \
                                      [0, 1, 0, 0, 1]]),\
                            np.array([[1, 0, 0, 1, 0], \
                                      [0, 0, 1, 0, 0], \
                                      [0, 1, 0, 0, 1]]),\
                            np.array([[0, 1, 0, 0, 0], \
                                      [1, 0, 1, 0, 1], \
                                      [0, 0, 0, 1, 0]]),\
                            np.array([[1, 1, 0, 0, 0], \
                                      [0, 1, 1, 1, 0], \
                                      [0, 0, 0, 1, 1]]) ]

    elif N == 6:
        flicker_mat_vec = [ np.array([[1, 0, 0, 0, 1, 0],\
                                      [0, 1, 0, 0, 0, 1],\
                                      [0, 0, 1, 1, 0, 0]]),\
                            np.array([[1, 1, 0, 0, 0, 0], \
                                      [0, 0, 1, 1, 0, 0], \
                                      [0, 0, 0, 0, 1, 1]]),\
                            np.array([[1, 0, 0, 1, 0, 0], \
                                      [0, 1, 1, 1, 1, 0], \
                                      [0, 0, 1, 0, 0, 1]]),\
                            np.array([[1, 0, 1, 0, 1, 0], \
                                      [0, 1, 0, 1, 0, 1], \
                                      [1, 1, 1, 1, 1, 1]]),\
                            np.array([[0, 1, 0, 0, 0, 0], \
                                      [1, 0, 1, 1, 0, 1], \
                                      [0, 0, 0, 0, 1, 0]]) ]



    freq_dict = {}
    score_dict = {}
    for i in range(len(flicker_mat_vec)):
        freq_dict[i] = []
        score_dict[i] = []

    for epoch in range(epochs):
        freq = min_f + np.random.rand() * (max_f-min_f)
        f_fun = lambda t: freq * np.sin(freq * 2 * np.pi * t)
        print('epoch =', epoch)
        #### True Signal #####
        sigTrue = SignalClass(np.array(f_fun(super_time_vec[:-int(1 / super_time_step)])),
                              time_step=super_time_step / camera_fps, name='True Signal')

        #### N #####
        b_vec, g_vec, r_vec = flicker_mat_vec[ epoch % len(flicker_mat_vec) ]
        signal_vec_x = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        # Temporal up-sampling:
        signal_vec_x = temporal_dense_vec(signal_vec_x, int(1 / (N * super_time_step)))
        sig4x = SignalClass(np.array(signal_vec_x), time_step=super_time_step / camera_fps, name='x'+str(N))

        # Compare perforemence:
        l2_scores = sigTrue.l2_distance(sig4x, int(0.5 / (N * super_time_step)))

        freq_dict[ epoch % len(flicker_mat_vec) ] += [freq]
        score_dict[ epoch % len(flicker_mat_vec) ] += [l2_scores]

    for i in range(len(flicker_mat_vec)):
        plt.plot(np.array(freq_dict[i]), np.array(score_dict[i]), 'o')
    plt.legend(range(len(flicker_mat_vec)))
    plt.title('L2 Score')
    plt.show()

    return

def compare_x1_x3_x4_flicker_random_vec(epochs, min_f, max_f, super_time_vec, super_time_step):
    def generate_flicker_vec_randomly(N):
        flag = True
        while flag:
            mat = np.int8(np.random.rand(3,N)+0.5)
            if np.linalg.matrix_rank(mat) == 3:
                flag = False
        b_vec = mat[0,:]
        g_vec = mat[1,:]
        r_vec = mat[2,:]
        return (b_vec, g_vec, r_vec)

    total_freq_vec = []
    total_l2_score_vec = []
    for epoch in range(epochs):
        freq = min_f + np.random.rand() * (max_f-min_f)
        f_fun = lambda t: freq * np.sin(freq * 2 * np.pi * t)
        print('epoch =', epoch)
        #### True Signal #####
        sigTrue = SignalClass(np.array(f_fun(super_time_vec[:-int(1 / super_time_step)])),
                              time_step=super_time_step / camera_fps, name='True Signal')
        #### N = 1 #####
        signal_vec_x1 = sample_function(f_fun, time_vec)
        #### N = 3 #####
        b_vec, g_vec, r_vec = get_bgr_vectors_from_N(3)
        signal_vec_x3 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)
        #### N = 4 #####
        b_vec, g_vec, r_vec = generate_flicker_vec_randomly(4)
        signal_vec_x4 = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

        # Temporal up-sampling:
        signal_vec_x1 = temporal_dense_vec(signal_vec_x1, int(1 / (1 * super_time_step)))
        signal_vec_x3 = temporal_dense_vec(signal_vec_x3, int(1 / (3 * super_time_step)))
        signal_vec_x4 = temporal_dense_vec(signal_vec_x4, int(1 / (4 * super_time_step)))
        sig1x = SignalClass(np.array(signal_vec_x1), time_step=super_time_step / camera_fps, name='x1')
        sig3x = SignalClass(np.array(signal_vec_x3), time_step=super_time_step / camera_fps, name='x3')
        sig4x = SignalClass(np.array(signal_vec_x4), time_step=super_time_step / camera_fps, name='x4')

        # Compare perforemence:
        l2_scores = [sigTrue.l2_distance(sig1x, int(0.5 / (super_time_step))), \
                     sigTrue.l2_distance(sig3x, int(0.5 / (3 * super_time_step))), \
                     sigTrue.l2_distance(sig4x, int(0.5 / (4 * super_time_step)))]

        total_freq_vec += [[freq] * 3]
        total_l2_score_vec += [l2_scores]

    plt.plot(np.array(total_freq_vec), np.array(total_l2_score_vec), '.')
    plt.legend(['x1', 'x3', 'x4 - random flicker'])
    plt.title('L2 Score')
    plt.show()

def evaluate_xN_score(N, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, white_noise_filter = 0.0, name =''):
    # This function gets N up sampling factor and fun function of the signal. It returns the up-sampling method result.

    #### N #####
    b_vec, g_vec, r_vec = get_bgr_vectors_from_N(N)

    true_signal_vec = np.array(f_fun(super_time_vec))
    signal_vec_x1 = sample_function(f_fun, time_vec)
    signal_vec_xN = simulate_rgb_sample(f_fun, time_vec, b_vec, g_vec, r_vec)

    #### N = 1 #####
    signal_vec_x1 = temporal_dense_vec(signal_vec_x1, int(1 / (1 * super_time_step)))

    ##### N ####
    signal_vec_xN = temporal_dense_vec(signal_vec_xN, int(1 / (N * super_time_step)))

    #### True Signal #####
    sigTrue = SignalClass(true_signal_vec, time_step=super_time_step / camera_fps, name='True Signal')
    sig1x = SignalClass(np.array(signal_vec_x1), time_step=super_time_step / camera_fps, name='x1')
    sigNx = SignalClass(np.array(signal_vec_xN), time_step=super_time_step / camera_fps, name='x'+str(N))

    sig1x.time_vec = sig1x.time_vec[int(0.5 / (1 * super_time_step)):]
    sig1x.signal = sig1x.signal[:-int(0.5 / (1 * super_time_step))]
    sigNx.time_vec = sigNx.time_vec[int(0.5 / (N * super_time_step)):]
    sigNx.signal = sigNx.signal[:-int(0.5 / (N * super_time_step))]
    sig1x.set_signal(sig1x.signal)
    sigNx.set_signal(sigNx.signal)

    sig1x.filter_white_noise(white_noise_filter)
    sigNx.filter_white_noise(white_noise_filter)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(sigTrue.frequencies, np.abs(sigTrue.signal_fft), '-')
    plt.plot(sig1x.frequencies, np.abs(sig1x.signal_fft), '-')
    plt.plot(sigNx.frequencies, np.abs(sigNx.signal_fft), 'r-')
    plt.legend(['True signal', 'x1', 'x'+str(N)])
    plt.title('Spectrum Comparison - Simulation: '+name)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Signal')
    plt.xlim([-30.0,30.0])
    plt.show()

if __name__ == '__main__':
    T = 10.0
    camera_fps = 10
    time_vec = np.arange(0,T,1./camera_fps)

    #### True Signal #####
    super_time_step = 1/6000
    super_time_vec = np.arange(0.0,T,super_time_step/camera_fps)
    shift = int((0.5/camera_fps) / (0.001/camera_fps))

    #### Random S comparison ####
    compare_x1_x3_x4_flicker_random_vec(5000, 5, 30, super_time_vec, super_time_step)
    exit()
    ##### Fix S comparison #####
    compare_fixed_pattern_flicker_vec(10000, 5, 30, 4, super_time_step)
    compare_fixed_pattern_flicker_vec(10000, 5, 30, 5, super_time_step)
    compare_fixed_pattern_flicker_vec(10000, 5, 30, 6, super_time_step)

    ##### N methods comparison #####
    compare_x1_x3_x4_x5_x6_score(10000, 5, 30, super_time_step)

    ##### Different N comparison examples #####
    noise_threshold = 0.0
    f_fun = lambda t: np.sin(1*2*np.pi*t) + np.sin(6*2*np.pi*t) + np.sin(11*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '1_6_11')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '1_6_11')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '1_6_11')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '1_6_11')
    f_fun = lambda t: np.sin(7*2*np.pi*t) + np.sin(12*2*np.pi*t) + np.sin(17*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '7_12_17')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '7_12_17')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '7_12_17')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '7_12_17')
    f_fun = lambda t: np.sin(3*2*np.pi*t) + np.sin(11*2*np.pi*t) + np.sin(23*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '3_11_23')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '3_11_23')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '3_11_23')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '3_11_23')
    f_fun = lambda t: np.sin(7*2*np.pi*t) + np.sin(28*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '7_28')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '7_28')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '7_28')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, '7_28')
    f_fun = lambda t: signal.square(5*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_5')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_5')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_5')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_5')
    f_fun = lambda t: signal.square(10*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_10')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_10')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_10')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_10')
    f_fun = lambda t: signal.square(15*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_15')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_15')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_15')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_15')
    f_fun = lambda t: signal.square(20*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_20')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_20')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_20')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_20')
    f_fun = lambda t: signal.square(25*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_25')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_25')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_25')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_25')
    f_fun = lambda t: signal.square(30*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_30')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_30')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_30')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_30')
    f_fun = lambda t: signal.square(5*2*np.pi*t) + signal.square(10*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_5_10')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_5_10')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_5_10')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_5_10')
    f_fun = lambda t: signal.square(11*2*np.pi*t) + signal.square(27*2*np.pi*t)
    evaluate_xN_score(3, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_11_27')
    evaluate_xN_score(4, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_11_27')
    evaluate_xN_score(5, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_11_27')
    evaluate_xN_score(6, f_fun, super_time_step, camera_fps, time_vec, super_time_vec, noise_threshold, 'square_11_27')
