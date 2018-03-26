import ftplib
import os

remote_host = "119.29.161.49"

pretrain_model_file = "VGG_imagenet.npy"
trainset_file = "icpr_text_train.zip"


def download():
    try:
        ftp = ftplib.FTP(remote_host)
    except:
        print("ERROR cannot reach '%s'" % remote_host)
        return
    print("..Connected to remote_host '%s'.." % remote_host)

    try:
        ftp.login()  # 使用匿名账号登陆也就是anonymous
    except ftplib.error_perm:
        print("ERROR cannot login anonymously")
        ftp.quit()
        return
    print("...logged in as 'anonymously'...")

    bufsize = 1024
    download_path = os.path.join('dataset/pretrain/')
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    file_handler = open(download_path + pretrain_model_file,'wb').write #以写模式在本地打开文件
    ftp.retrbinary('RETR %s' % pretrain_model_file , file_handler, bufsize)  # 接收服务器上文件并写入本地文件

    ftp.quit()
    return


# 调用函数执行测试
if __name__ == "__main__":
    download()