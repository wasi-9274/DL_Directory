import re
import dns.resolver
import socket
import smtplib


# email_list = ['wasi@gmail.com',
#               'ahmed@gmail.com',
#               'wasiahmed@gmail.com',
#               'wasiahmed9274@gmail.com'
#               'wasi.ahmed@gmail.com',
#               'wahmed@gmail.com',
#               'w.ahmed@gmail.com',
#               'wasia@gmail.com',
#               'wasi.a@gmail.com',
#               'wa@gmail.com',
#               'w.a@gmail.com',
#               'ahmedwasi@gmail.com',
#               'ahmed.wasi@gmail.com',
#               'ahmedw@gmail.com',
#               'ahmed.w@gmail.com',
#               'awasi@gmail.com',
#               'a.wasi@gmail.com',
#               'aw@gmail.com',
#               'a.w@gmail.com',
#               'wasi-ahmed@gmail.com',
#               'w-ahmed@gmail.com',
#               'wasi-a@gmail.com',
#               'w-a@gmail.com',
#               'ahmed-wasi@gmail.com',
#               'ahmed-w@gmail.com',
#               'a-wasi@gmail.com',
#               'a-w@gmail.com',
#               'wasi_ahmed@gmail.com',
#               'w_ahmed@gmail.com',
#               'wasi_a@gmail.com',
#               'w_a@gmail.com',
#               'ahmed_wasi@gmail.com',
#               'ahmed_w@gmail.com',
#               'a_wasi@gmail.com',
#               'a_w@gmail.com']

email_list = ['wasiahmed9274@gmail.com']

for e in email_list:
    addressToVerify = e
    match = re.match('^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$', addressToVerify)

    if match == None:
        print('Bad Syntax')
        raise ValueError('Bad Syntax')


    records = dns.resolver.query('emailhippo.com', 'MX')
    mxRecord = records[0].exchange
    mxRecord = str(mxRecord)

    host = socket.gethostname()

    server = smtplib.SMTP()
    server.set_debuglevel(0)

    server.connect(mxRecord)
    server.helo(host)
    server.mail('me@domain.com')
    code, message = server.rcpt(str(addressToVerify))
    server.quit()

    if code == 250:
        print('Success for the email address -> {} and Code is -> {}'.format(addressToVerify, code))
    else:
        print('Might not exist -> {} and Code for failure is {}'.format(addressToVerify, code))
