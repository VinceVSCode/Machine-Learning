import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


"""
     Sxolia peri kwdika ftiaxnoume ena pinaka 24 thesewn pou antiprosopeuei tis
   wres mias hmeras epeita ena pinaka me tuxaies times apo to 0~100 pou antiorisopeuoun 
   thn piothta parakatw vlepoume ton tupo ths aplhs grammikhs palindromishs.
   Simple Linear regression prediction 𝑓(𝑥) = 𝑏₀ + 𝑏₁𝑥
   To programma friaxnei to montelo apo ta dedomena kai kanei provlepseis.Logo 
   twn tuxaiwn dedomenwn mporei na uparxei overfitting h underfitting.
   Ftiaxnoume kai ena diagramma gia kaluterh katanohsh.

"""

# Create our axis data
x_axis = np.array([(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24)]).reshape((-1, 1))  # hours a day
y_axis = np.random.random_integers(0, 100, size=(24, 1))  # quality of air

print(x_axis, "\n", y_axis)

# Import model
model = LinearRegression()

# Train model
model.fit(x_axis, y_axis)

R_2 = model.score(x_axis, y_axis)
print("Coefficient of determination or R^2 is => ", R_2)

print("Intercept b0 => ", model.intercept_)
print("Slope or b1", model.coef_)

plt.scatter(x_axis, y_axis)
plt.plot(x_axis, model.coef_ * x_axis + model.intercept_)
plt.title("Time and Estimated Air Quality", fontweight="bold")
plt.xlabel("Time")
plt.ylabel("Estimated Air quality")
plt.show()

# Make Predictions
while True:
    say = np.array(
        [((float(input("Give me the time number I will predict. \n"))))])
    if say > 24 or say < 0:
        print("Give me time between 0-24")
        continue
    y_predict = model.predict(say.reshape((-1, 1)))
    print("For ", say, " the predicted value is ->", y_predict)
    if (input("Want to stop? y/n \n") == 'y'):
        break
