import os 

with open("icarus-cdk-danner/local-files/campaign_advisor_prompt.md", "r") as f:
    temp = f.read()

temp = temp.replace("{candidate_questionnaire}", "HELLO WORLD")

print(temp)