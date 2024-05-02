import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

class FeatureSelectionABC:
    def __init__(self, num_bees, max_iter, limit=5, position=None, score=None, trial=0):
        self.position = position if position is not None else np.array([])  # Initialize position if not provided
        self.score = score
        self.trial = trial
        self.num_bees = num_bees
        self.max_iter = max_iter
        self.limit = limit

    def initialize_bees(self, num_features):
        bees = []
        for _ in range(self.num_bees):
            position = np.random.randint(2, size=num_features)
            bees.append(FeatureSelectionABC(self.num_bees, self.max_iter, self.limit, position))
        return bees

    def calculate_fitness(self, X_train, X_test, y_train, y_test):
        selected_features = np.where(self.position == 1)[0]
        if len(selected_features) == 0:
            return 0
        X_train_selected = X_train.values[:, selected_features]
        X_test_selected = X_test.values[:, selected_features]

        classifier = GaussianNB()
        classifier.fit(X_train_selected, y_train)
        predictions = classifier.predict(X_test_selected)
        score = accuracy_score(y_test, predictions)
        return score

    def employed_bees_phase(self, bees, X_train, X_test, y_train, y_test):
        for bee in bees:
            new_position = np.copy(bee.position)
            phi = np.random.uniform(-1, 1, size=len(bee.position))
            partner_index = np.random.randint(0, len(bees))
            new_position = np.clip(new_position + phi * (new_position - bees[partner_index].position), 0, 1).astype(int)

            new_bee = FeatureSelectionABC(self.num_bees, self.max_iter, self.limit,new_position)
            new_bee.score = new_bee.calculate_fitness(X_train, X_test, y_train, y_test)

            if new_bee.score > bee.score:
                bee.position = new_bee.position
                bee.score = new_bee.score
                bee.trial = 0
            else:
                bee.trial += 1
                
    def onlooker_bees_phase(self, bees, X_train, X_test, y_train, y_test):
        sum_scores = sum([bee.score for bee in bees])
        probs = [bee.score / sum_scores for bee in bees]

        for _ in range(len(bees)):
            bee_index = np.random.choice(range(len(bees)), p=probs)
            bee = bees[bee_index]
            self.employed_bees_phase([bee], X_train, X_test, y_train, y_test)

    def scout_bees_phase(self, bees, num_features):
        for bee in bees:
            if bee.trial > self.limit:
                bee.position = np.random.randint(2, size=num_features)
                bee.trial = 0

    def optimize_bees(self, X_train, X_test, y_train, y_test):
        num_features = X_train.shape[1]
        bees = self.initialize_bees(num_features)

        for _ in range(self.max_iter):
            for bee in bees:
                bee.score = bee.calculate_fitness(X_train, X_test, y_train, y_test)

            self.employed_bees_phase(bees, X_train, X_test, y_train, y_test)
            self.onlooker_bees_phase(bees, X_train, X_test, y_train, y_test)
            self.scout_bees_phase(bees, num_features)

        best_bee = max(bees, key=lambda b: b.score if b.score is not None else 0)
        return best_bee.position, best_bee.score

