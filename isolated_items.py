import pandas
import numpy
import math

WEEKEND_NIGHTS = 'stays_in_weekend_nights'
WEEK_NIGHTS = 'stays_in_week_nights'
ADULTS = 'adults'
CHILDREN = 'children'
BABIES = 'babies'


df = pandas.read_csv("hotel_bokings_2017.csv", sep=",")
print(df.columns)


labels = [WEEKEND_NIGHTS, WEEK_NIGHTS, ADULTS, CHILDREN, BABIES]

labels_mean_std = {}
with open("isolated.dat", "w") as file:
    for label in labels:
        mean = df[label].mean()
        std = df[label].std()
        labels_mean_std[label] = [mean, std, df[label].sum()]

    file.write("=== Isolated cases ===\n")
    for label in labels:
        limit = 3 * labels_mean_std[label][1]
        mean = labels_mean_std[label][0]
        file.write(label + " Mean: " + str(mean) + " std: " + str(labels_mean_std[label][1]) + " limit: " + str(limit) + "\n")

        for i in range(len(df[label])):
            if numpy.fabs(df[label][i] - mean) > limit:
                file.write("\t" + str(df[label][i]) + " " + df["hotel"][i] + " " + str(df["arrival_date_month"][i]) + " " + str(df["arrival_date_day_of_month"][i]) + '\n')

        file.write("\n\n")



days_spend_mean =  (df[WEEK_NIGHTS].sum() + df[WEEKEND_NIGHTS].sum())/len(df[WEEK_NIGHTS])
persons_mean = (df[ADULTS].sum() + df[CHILDREN].sum() + df[BABIES].sum()) / len(df[ADULTS])
days_spend_sdt = 0
persons_std = 0

for i in range(len(df)):
    days_spend_sdt += ((df[WEEK_NIGHTS][i] + df[WEEKEND_NIGHTS][i])/days_spend_mean) ** 2
    persons_std += ((df[ADULTS][i] + df[CHILDREN][i] + df[BABIES][i])/persons_mean) ** 2

days_spend_sdt = math.sqrt(days_spend_sdt/len(df[WEEK_NIGHTS]))
persons_std = math.sqrt(persons_std/len(df[ADULTS]))

with open("Isolated person and days.dat", "w") as file:
    file.write(
        "Days" + " Mean: " + str(days_spend_mean) + " std: " + str(days_spend_sdt) + "\n")

    limit = 3 * days_spend_sdt
    for i in range(len(df)):
        days = df[WEEK_NIGHTS][i] + df[WEEKEND_NIGHTS][i]
        if numpy.fabs(days - days_spend_mean) > limit:
            file.write(
                "\t" + str(days) + " " + df["hotel"][i] + " " + str(df["arrival_date_month"][i]) + " " + str(
                    df["arrival_date_day_of_month"][i]) + '\n')

    file.write("\n\n")
    file.write(
        "Person" + " Mean: " + str(persons_mean) + " std: " + str(persons_std) + "\n")

    limit = 3 * persons_std
    for i in range(len(df)):
        persons = df[ADULTS][i] + df[CHILDREN][i] + df[BABIES][i]
        if numpy.fabs(persons - persons_mean) > limit:
            file.write(
                "\t" + str(persons) + " " + df["hotel"][i] + " " + str(df["arrival_date_month"][i]) + " " + str(
                    df["arrival_date_day_of_month"][i]) + '\n')




