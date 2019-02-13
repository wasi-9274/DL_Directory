import pandas as pd
import sqlalchemy
import os
from datetime import datetime
import shutil


engine = sqlalchemy.create_engine('mysql+pymysql://root:password@192.168.0.87:3306/contact_crm_data')
engine.connect()

for (root, dirs, files) in os.walk('/home/wasi/work_space/test'):
    for name in files:
        filename = os.path.join(root, name)
        print("=" * 85)
        print(name)
        print("=" * 85)
        df = pd.read_csv(filename, encoding='utf-8')

        if "DUNS Number" in df.columns:
            print("Found with the Condition 'DUNS Number' changing it to 'DUNS_Number'")
            df.rename(columns = {'DUNS Number': 'DUNS_Number'}, inplace=True)

        if "SIC_Code Description" in df.columns:
            print("Found with the Condition 'SIC_Code Description' changing it to 'SIC_Code_Description'")
            df.rename(columns = {'SIC_Code Description': 'SIC_Code_Description'}, inplace=True)

        with engine.connect() as connect:
            try:
                df.to_sql('contact_data', engine, if_exists='append', index=False)
                print(" Successfully stored into database : %s", format(str(datetime.time(datetime.now()))))
            except Exception as e:
                print('Some exception occured! -> {}'.format(e))
                print("Moving the -> {} file...".format(name))
                try:
                    shutil.move(filename, '/home/wasi/work_space/re_check_copied_files')
                except Exception as e:
                    print("Exception Occured -> {}".format(e))
        connect.close()
    engine.dispose()