# airline_passenger_satisfaction


## How to install dependencies

Declare any dependencies in `src/requirements.txt`.

To install them, run:

```
pip install -r src/requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.

## The origin of the DataSet is:
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction


## Create your credentials environment variables
```
export AWS_ACCESS_KEY_ID=****** AWS_SECRET_ACCESS_KEY=******
```

## Package the models with docker

```
kedro docker build --docker-args "--no-cache --build-arg AWS_ACCESS_KEY=$AWS_ACCESS_KEY --build-arg AWS_SECRET_KEY_ID=$AWS_SECRET_KEY_ID"
```

## and run it
```
docker run -d -p 5001:5001 airline-passenger-satisfaction
```

## Exemple of request data to test your API
{
    "gender": "Male",
    "customer_type": "Loyal Customer",
    "age": 13,
    "type_of_travel": "Personal Travel",
    "flight_class": "Eco Plus",
    "flight_distance": 460,
    "inflight_wifi_service": 3,
    "departure_arrival_time_convenient": 4,
    "ease_of_online_booking": 3,
    "gate_location": 1,
    "food and drink": 5,
    "online boarding": 3,
    "seat_comfort": 5,
    "inflight_entertainment": 5,
    "on_board_service": 4,
    "leg_room_service": 3,
    "baggage_handling": 4,
    "checkin_service": 4,
    "inflight_service": 5,
    "cleanliness": 5,
    "departure_delay_in_minutes": 25,
    "arrival_delay_in_minutes": 18.0
}
