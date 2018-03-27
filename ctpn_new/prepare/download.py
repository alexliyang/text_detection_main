# import ftplib
# import os
#
# remote_host = "119.29.161.49"
#
pretrain_model_file = "VGG_imagenet.npy"
# trainset_file = "icpr_text_train.zip"
#
#
# def download():
#     try:
#         ftp = ftplib.FTP(remote_host)
#     except:
#         print("ERROR cannot reach '%s'" % remote_host)
#         return
#     print("..Connected to remote_host '%s'.." % remote_host)
#
#     try:
#         ftp.login()  # 使用匿名账号登陆也就是anonymous
#     except ftplib.error_perm:
#         print("ERROR cannot login anonymously")
#         ftp.quit()
#         return
#     print("...logged in as 'anonymously'...")
#
#     bufsize = 1024
#     download_path = os.path.join('dataset/pretrain/')
#     if not os.path.exists(download_path):
#         os.makedirs(download_path)
#
#     file_handler = open(download_path + pretrain_model_file,'wb').write #以写模式在本地打开文件
#     ftp.retrbinary('RETR %s' % pretrain_model_file , file_handler, bufsize)  # 接收服务器上文件并写入本地文件
#
#     ftp.quit()
#     return
#
#
# # 调用函数执行测试
# if __name__ == "__main__":
#     download()

import requests
url = 'https://doc-0o-14-docs.googleusercontent.com/docs/securesc/73al6pct57n4oeq4vf5bq1jgdd34vlee/9o46t1mpu8k1okk09pcq30tv6t2mc6qj/1522108800000/09469196320027156440/09613593125903115016/0B_WmJoEtfQhDYTRvNU9PQmlydlk?e=download&nonce=rkh7crn2dij0k&user=09613593125903115016&hash=ain5kp2r2t1045ofguf7tti9h4dnp3fn'
r = requests.get(url)
print('get url successful')
with open(pretrain_model_file, "wb") as code:
    code.write(r.content)