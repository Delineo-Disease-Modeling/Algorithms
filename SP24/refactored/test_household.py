import yaml



class Person:
    _last_id = 0
    def __init__(self, age:int, sex:int, hh_id:int, tags:dict, cbg:int = None):
        """
        Initializes the Person class
        """
        self.id = Person._last_id
        Person._last_id += 1
        self.age = age
        self.sex = sex
        self.hh_id = hh_id
        self.tags = tags
        self.cbg = cbg

    def to_dict(self):
        return {
            "id": self.id,
            "age": self.age,
            "sex": self.sex
            "hh_id": self.hh_id,
            "tags": self.tags,
            "cbg": self.cbg
        }





def read_population_info():
    with open("input/population_info.yaml", "r") as file:
        pop_info = yaml.load(file, Loader=yaml.FullLoader)
        return pop_info



if __name__ == "__main__":
    # read population info
    pop_info = read_population_info()
