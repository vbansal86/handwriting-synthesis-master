import pandas as pd
import argparse
import numpy as np
from hwprescription import HWPrescription
from faker import Faker
fake = Faker()

handwrittenprescription = HWPrescription()


drugs_df = pd.read_csv('drug.csv')
drugs_df.columns = ['nbr', 'Seq', 'Practice', 'firstName', 'lastName', 'Address', 'BNFName']
unique_drugs_df = drugs_df.BNFName.unique()


def chunkstring(string, length):
    return (string[0+i:length+i] for i in range(0, len(string), length))

def get_random_drug():
    
    originalBNFName =  unique_drugs_df[np.random.randint(len(unique_drugs_df))]
    druglist = list(chunkstring(originalBNFName, 40))
    drug = druglist[np.random.randint(2)].rstrip()
    
    return {'originalBNFName': originalBNFName, 'drug': drug }
    
def get_prescriber (drug):
    temp_df = pd.DataFrame(data=drugs_df.loc[drugs_df['BNFName'] == drug])
    return temp_df.iloc[[np.random.randint(len(temp_df))]]


    

def generate_prescriptions (num, bias, style) :


    def clean_untrained_letters(word) :
        word = word.replace("_", " ")
        word = word.replace("Z", "z")
        word = word.replace("X", "x")
        word = word.replace("Q", "q")
        word = word.replace("%", " ")
        word = word.replace("#", " ")
        word = word.replace("@", " ")
        word = word.replace("&", " ")
        return word
    
            
            
    def generate_prescription(num, bias, style):
        randomPrescription = get_random_prescription()
        fileName = "Prescription" + str(num) + ".png"

        randomPrescription['prescriber_firstName'] = clean_untrained_letters (randomPrescription['prescriber_firstName'])
        randomPrescription['prescriber_lastName'] = clean_untrained_letters (randomPrescription['prescriber_lastName'])
        randomPrescription['prescriber_Address'] = clean_untrained_letters (randomPrescription['prescriber_Address'])
        randomPrescription['drug'] = clean_untrained_letters (randomPrescription['drug'] )
        
        

        
        lines = []
        lines.append(randomPrescription['prescriber_firstName'] + " " + randomPrescription['prescriber_lastName'])
        lines.append(randomPrescription['prescriber_Address'])
        lines.append(randomPrescription['drug'])

        biases = [bias for i in lines]
        styles = [style for i in lines]
        
        handwrittenprescription.write(
            filename=fileName,
            lines=lines,
            biases=biases,
            styles=styles
        )
        
        print (randomPrescription)

    def get_random_prescription():
        randomdrug = get_random_drug()
        prescriber = get_prescriber(randomdrug['originalBNFName'])
        patient_name = fake.name()
        
        
        return {'prescriber_firstName': prescriber.firstName.to_string(index=False),
                'prescriber_lastName': prescriber.lastName.to_string(index=False),
                'prescriber_Practice': prescriber.Practice.to_string(index=False),
                'prescriber_Address': prescriber.Address.to_string(index=False),
                'patient_name': patient_name,
                'originalBNFName': randomdrug['originalBNFName'],
                'drug': randomdrug['drug'] }
    
    for cnt in range (num) :
        generate_prescription (cnt, bias, style)
        


if __name__== '__main__':
    parser = argparse.ArgumentParser(description="Generate fictional prescriptions")
    parser.add_argument('--n', type=int, default=5, help='Number of prescriptions to generate')
    parser.add_argument('--bias', type=float, default=0.75, help='biases in handwriting')
    parser.add_argument('--style', type=int, default=10, help='styles in handwriting')

    args, unknown = parser.parse_known_args()
    args = vars(args)

    generate_prescriptions (args.get("n"), args.get("bias"), args.get("style"))
