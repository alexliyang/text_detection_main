import ctpn.train_net as train_net
import pprint

if __name__ == '__main__':
    cfg_from_file('ctpn/text.yml')
    print('Using config:')
    pprint.pprint(cfg)

    imdb = get_imdb()

    roidb = get_training_roidb(imdb)

    output_dir = ''
    log_dir = ''
    print('Output will be saved to `{:s}`'.format(output_dir))

    print('Logs will be saved to `{:s}`'.format(log_dir))



    network = get_network('VGGnet_train')
    
    
    train_net()