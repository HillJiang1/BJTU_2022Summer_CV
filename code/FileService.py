import pandas as pd

def updatePeopleInfo(id,name,type):
    newUser =[[id,name,type]]
    df = pd.DataFrame(newUser)
    print(newUser)
    df.to_csv("../info/people_info.csv", sep=',', mode='a', index= False, header= False)
    print('update')
