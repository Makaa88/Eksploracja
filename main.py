import pandas
import sklearn

df = pandas.read_csv("hotel_bookings.csv")

list_to_save = []
days_spend = []
all_guests = []

for i in range(len(df)):
    if df["arrival_date_year"][i] == 2016:
        list_to_save.append(df.iloc[i])
        days_spend.append(df["stays_in_weekend_nights"][i] + df["stays_in_week_nights"][i])
        all_guests.append(df["adults"][i] + df["children"][i] + df["babies"][i])

df_to_save = pandas.DataFrame(list_to_save, columns=df.columns)
df_to_save["days_in_hotel"] = days_spend
df_to_save["all_guests"] = all_guests

df_to_save.to_csv("hotel_bokings_2016.csv", sep=',')