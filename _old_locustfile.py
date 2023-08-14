from locust import HttpUser, TaskSet, task, between


def recos(l):
    l.client.post(url='/v1/recommend', json={"item_ids": [1, 5, 3, 1, 7, 1], "session_id": "abcdefg"})


class UserTasks(TaskSet):
    # one can specify tasks like this
    tasks = [recos]


class WebsiteUser(HttpUser):
    """
    User class that does requests to the locust web server running on localhost
    """

    host = "http://127.0.0.1:8089"
    tasks = [UserTasks]
