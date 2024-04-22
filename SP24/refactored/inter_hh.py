from household import Household, Person
import pandas as pd
import numpy as np

class InterHousehold:
    def __init__(self, hh_list:list[Household]):
        self.iteration = 1
        self.hh_list:list[Household] = np.array(hh_list)
        self.people_list  = [] # currently not used


        # we create a map that map person id to the person object
        self.id_to_person:dict[int, Person] = {}
        self.id_to_household:dict[int, Household] = {}

        people_data:list[dict] = []
        household_data:list[dict] = []
        for hh in hh_list:
            household_data.append(hh.to_dict())
            # map id to household
            self.id_to_household[hh.id] = hh
            for p in hh.population:
                people_data.append(p.to_dict())
                # map id to person
                self.id_to_person[p.id] = p
                self.people_list.append(p)
        
        self.people_df = pd.DataFrame(people_data)
        self.household_df = pd.DataFrame(household_data)
        self.people_list:list[Person] = np.array(self.people_list)

        self.verbose = True
        
        self.social_hh:set[Household] = set()
        self.movement_people:set[Person] = set()
        self.children:set[tuple[Person, Person]] = set() # (child, parent)

        self.individual_movement_frequency = 0.2

        self.social_event_frequency = 0.05
        self.social_guest_num = 4
        self.social_max_duration = 3
        self.social_event_hh_cap = 0.1 # percentage of the population

        self.school_children_frequency = 0.18



        self.regular_visitation_frequency = 0.15

        self.prefer_cbg = 0.7 # possibility that guests come from the same cbg

    def update_df(self):
        # update availability
        self.people_df['availability'] = self.people_df['id'].map(lambda x: self.id_to_person[x].availability)
        self.people_df['at_home'] = self.people_df['id'].map(lambda x: self.id_to_person[x].at_home()) 


        '''
        self.people_df[['availability', 'at_home']] = self.people_df['id'].apply(lambda x: pd.Series({
            'availability': self.id_to_person.get(x).availability,
            'at_home': self.id_to_person.get(x).at_home(),
        }))
        '''
    
    

    def filter_df(self, df:pd.DataFrame, **kwargs) -> pd.DataFrame:
        for key, (operation, value) in kwargs.items():
            if key not in df.columns:
                raise KeyError(f"Column {key} not found in DataFrame.")
            
            # Apply filter based on operation
            if operation == '==':
                df = df[df[key] == value]
            elif operation == '!=':
                df = df[df[key] != value]
            elif operation == '<':
                df = df[df[key] < value]
            elif operation == '<=':
                df = df[df[key] <= value]
            elif operation == '>':
                df = df[df[key] > value]
            elif operation == '>=':
                df = df[df[key] >= value]
            elif operation == 'in':
                if not isinstance(value, list):
                    raise ValueError("For 'in' operation, value must be a list.")
                df = df[df[key].isin(value)]
            else:
                raise ValueError(f"Unsupported operation {operation}")
        
        return df

    

    def select_population_by_probability(self, df:pd.DataFrame, probability:float, mode:str, **kwargs) -> list[Household] | list[Person]:
        """
        Filters the DataFrame based on provided keyword arguments using advanced comparison operations and selects a population by probability.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the population data.
            probability (float): The probability of selecting each entry.
            mode (str): Mode of mapping; 'h' for household and 'p' for person.
            **kwargs: Keyword arguments specifying filtering criteria where keys are column names and values are tuples containing (operation, value).

        Returns:
            list: Mapped values of the selected population based on the mode.

        Raises:
            ValueError: If mode is not 'h' or 'p' or if probability is not between 0 and 1.
            KeyError: If a key specified in kwargs does not exist in the DataFrame.
        """

        if mode not in ['h', 'p']:
            raise ValueError("Mode must be 'h' (household) or 'p' (person).")
        if not (0 <= probability <= 1):
            raise ValueError("Probability must be between 0 and 1.")
            
        df = self.filter_df(df, **kwargs)

        df = df["id"]
        if mode == "h":
            return Random.mapped_population_by_threshold(df, self.id_to_household, probability)
        elif mode == "p":
            return Random.mapped_population_by_threshold(df, self.id_to_person, probability)

    def select_population_by_number(self, df:pd.DataFrame, num:int, mode:str, fallback=False, **kwargs) -> list[Household] | list[Person]:
        """
        Filters the DataFrame based on provided keyword arguments and selects a fixed number of population entries using advanced comparison operations.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the population data.
            num (int): The number of entries to select.
            mode (str): Mode of mapping; 'h' for household and 'p' for person.
            fallback (bool): If true, select all available entries if `num` exceeds available entries.
            **kwargs: Keyword arguments specifying filtering criteria where keys are column names and values are tuples containing (operation, value).

        Returns:
            list: Mapped values of the selected population based on the mode.

        Raises:
            ValueError: If mode is not 'h' or 'p' or if num is negative or exceeds the number of available entries.
            KeyError: If a key specified in kwargs does not exist in the DataFrame.
        """

        if mode not in ['h', 'p']:
            raise ValueError("Mode must be 'h' (household) or 'p' (person).")
    
            
        df = self.filter_df(df, **kwargs)

        df = df["id"]
        
        if num < 0 or num > len(df):
            if fallback:
                num = len(df)
            else:
                raise ValueError("Number of entries to select must be a non-negative integer and cannot exceed the number of available entries.")

        if mode == "h":
            return Random.mapped_population_by_number(df, self.id_to_household, num)
        elif mode == "p":
            return Random.mapped_population_by_number(df, self.id_to_person, num)

        
    def random_boolean(self, probability_of_true:float):
        """
        Generates a random boolean value based on the given probability of True.
        
        :param probability_of_true: A float representing the probability of returning True, between 0 and 1.
        :return: A boolean value, where True occurs with the given probability.
        """
        return np.random.random() < probability_of_true


    def next(self):
        self.update_df()
        self.individual_movement()
        self.social_event()
        self.children_movement()
        print(f"InterHousehold iteration {self.iteration}")
        self.iteration += 1
        
        if self.verbose:
            hosts = 0
            guests = 0
            social = 0
            for p in self.people_list:
                if p.location.id == p.household.id:
                    hosts += 1
                elif p.location.is_social():
                    social += 1
                else:
                    guests += 1

            print(f"hosts: {hosts}, guests: {guests}, social: {social} ----- total: {hosts + guests + social}")


    
    def children_movement(self):
        # move children and parent to their own household
        for c, p in self.children: # (child, parent)
            c.assign_household(c.household)
            p.assign_household(p.household)
            

        children:list[Person] = self.select_population_by_probability(self.people_df, self.school_children_frequency, "p", age=('<', 12), availability=('==', True), at_home=('==', True))
        for child in children:
            parent = self.select_population_by_number(self.people_df, 1, "p", True, home=("==", child.household.id), age=('>=', 18), availability=('==', True), at_home=('==', True))
            if len(parent) > 0:
                parent = parent[0]
                # move the child with its parent
                # select destination household
                same_cbg = self.random_boolean(self.prefer_cbg)
                if same_cbg:
                    hh = self.select_population_by_number(self.household_df, 1, "h", True, cbg=("==", child.cbg), id=("!=", child.household.id))[0]
                else:
                    hh = self.select_population_by_number(self.household_df, 1, "h", True, id=("!=", child.household.id))[0]
                
                # assign to new houses
                child.assign_household(hh)
                parent.assign_household(hh)
                self.children.add((child, parent))



    def social_event(self):
        # increase the houses already in social by 1 day
        for hh in list(self.social_hh):
            if hh.social_days >= hh.social_max_duration: # if the host day reachees maximum, stop social
                hh.end_social()
                self.social_hh.discard(hh)
            else:
                hh.social_days += 1

        hh_social:list[Household] = self.select_population_by_probability(self.household_df, self.social_event_frequency, "h")

        if (len(self.social_hh) + len(hh_social) >= len(self.id_to_household)*self.social_event_hh_cap):
            return # current social event housings must be under the cap

        for hh in hh_social: # iterate through the randomly selected households
            if not hh.is_social() and hh.has_hosts(): # if the household is not hosting social and the household has its original population
                self.social_hh.add(hh)
                hh.start_social(duration=np.random.randint(1, self.social_max_duration + 1))

                # select guests
                guest_num = np.random.randint(1, self.social_guest_num + 1)
                same_cbg = self.random_boolean(self.prefer_cbg)
                if same_cbg:
                    guest = self.select_population_by_number(self.people_df, guest_num, "p", True, cbg=('==', hh.cbg), age=('>=', 18), at_home=('==', True))
                else:
                    guest = self.select_population_by_number(self.people_df, guest_num, "p", True, age=('>=', 18), at_home=('==', True))
                
                for person in guest:
                    if person.household.id != hh.id and not person.location.is_social() and person.availability: # if the person does not belong to the household member and is not in a social and person is availiable
                        person.assign_household(hh)


    def individual_movement(self):
        for person in self.movement_people:
            # put the person back to its original household
            person.assign_household(person.household)
        self.movement_people = set()

        selected_people:list[Person] = self.select_population_by_probability(self.people_df, self.individual_movement_frequency, "p", age=('>=', 18), at_home=('==', True))

        for person in selected_people:
            if not person.location.is_social() and person.availability: # if move and person is not in social and person is not going to work
                same_cbg = self.random_boolean(self.prefer_cbg)
                if same_cbg:
                    hh = self.select_population_by_number(self.household_df, 1, "h", False, cbg=('==', person.cbg))[0]
                else:
                    hh = self.select_population_by_number(self.household_df, 1, "h", False)[0]
                
                if (hh.id != person.location.id or not hh.is_social()):
                    person.assign_household(hh)
                    self.movement_people.add(person)


class Random:
    """
    Two types of people selction
    1. every element has equal probability of being selected. Generate a list of probaility for each element, select the element who's probability reaches the treshold.
    2. select a fixed number of sample from the pool
    """
    @staticmethod
    def mapped_population_by_threshold(population_id: list | np.ndarray | pd.Series, map_data: dict, probability: float) -> list:
        """
        Samples a list of population IDs based on a specified probability and maps each selected ID to its corresponding object using a provided dictionary. This method returns the list of objects after mapping.
        A probability is generated for each ID in the list, and any ID with probability smaller than the probability parameter will be included in the output.
        
        Parameters:
            population_id (list | np.ndarray | pd.Series): The IDs of the population to sample from.
            map_data (dict): Dictionary that maps each ID to its corresponding object.
            probability (float): Probability of selecting each ID.

        Returns:
            list: Mapped objects of the sampled population IDs.

        Raises:
            ValueError: If the probability is not between 0 and 1.
            TypeError: If population_id is neither an ndarray nor a pd.Series.
            KeyError: If a key from the population IDs is missing in map_data.
        """

        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        
        # Calculate indices of the population to be sampled
        index_list = Random.select_by_threshold(0, len(population_id), probability)
        
        try:
            if isinstance(population_id, pd.Series):
                sample = [map_data[population_id.iat[index]] for index in index_list]
            else:
                # Retrieve population data from index and map to new values using map_data
                sample = [map_data[population_id[index]] for index in index_list]


        except KeyError as e:
            raise KeyError(f"Missing key in map_data: {e}")
        
        return sample
    
    def mapped_population_by_number(population_id, map_data, num):
        """
        Samples a fixed number of population IDs and maps each selected ID to its corresponding object using a provided dictionary. 
        This method efficiently returns a list of objects corresponding to the mapped IDs.
        
        Parameters:
            population_id (list | np.ndarray | pd.Series): The IDs of the population to sample from.
            map_data (dict): Dictionary that maps each ID to its corresponding object.
            num (int): Fixed number of IDs to sample.
        
        Returns:
            list: Mapped objects of the sampled population IDs.
        
        Raises:
            TypeError: If the type of population_id is not supported.
            ValueError: If the number of elements to sample is out of an acceptable range or if num is not a non-negative integer.
            KeyError: If a key from the population IDs is missing in map_data.
        """

        if num == 0:
            return []

        if not isinstance(num, int) or num < 0:
            raise ValueError("Number of elements to sample must be a non-negative integer.")
        if num > len(population_id):
            raise ValueError("Number of elements to sample cannot exceed the size of the population.")

        try:
            # Sampling indices directly from population_id if it is a pd.Series, or using a random choice for arrays or lists
            if isinstance(population_id, pd.Series):
                sampled_ids = population_id.sample(n=num, replace=False)
            elif isinstance(population_id, (np.ndarray, list)):
                sampled_ids = np.random.choice(population_id, num, replace=False)
            else:
                raise TypeError("population_id must be a list, np.ndarray, or pd.Series.")
            
            # Mapping sampled IDs to objects
            mapped_objects = [map_data[_id] for _id in sampled_ids]

        except KeyError as e:
            raise KeyError(f"Missing key in map_data: {e}")

        return mapped_objects


    @staticmethod
    def select_by_threshold(start:int, end:int, probability:float) -> np.ndarray:
        """
        Select numbers from a specified range [start, end) with a given probability for each number.
        The end of the range is exclusive, similar to Python's range() function.

        Parameters:
            start (int): The starting number of the range (inclusive).
            end (int): The ending number of the range (exclusive).
            probability (float): The probability of each number being selected, between 0 and 1.

        Returns:
            np.ndarray: An array of selected numbers, each selected with the generated probability lower than the input probability.

        Example:
            >>> Random.select_by_threshold(0, 100, 0.2)
            array([3, 15, 20, ..., 95])  # Example output; actual output will vary
        """
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

        if end <= start:
            raise ValueError("End must be greater than start.")

        # Calculate the number of elements in the range
        range_size = end - start

        # Generate a random float for each number in the range and compare to the probability
        random_draws = np.random.rand(range_size) < probability

        # Extract the indices where the draws were successful (True), and adjust for the range start
        selected_numbers = np.where(random_draws)[0] + start

        return selected_numbers