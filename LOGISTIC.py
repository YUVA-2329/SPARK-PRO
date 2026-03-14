# FRUIT CLASSIFIER USING K-NEAREST NEIGHBORS (KNN)
# A simple machine learning program to classify fruits based on their characteristics

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import matplotlib.patches as mpatches

class FruitClassifier:
    def __init__(self):
        """Initialize the Fruit Classifier with a custom fruit dataset"""
        print("\n" + "🍎" * 50)
        print("     FRUIT CLASSIFIER USING K-NEAREST NEIGHBORS (KNN)")
        print("🍎" * 50)
        
        # Create our own fruit dataset
        # Features: [Weight (grams), Sweetness (1-10), Color (1-3), Size (1-5)]
        # Color: 1=Red, 2=Green, 3=Yellow
        self.create_fruit_dataset()
        
        # Initialize model
        self.model = None
        self.k_value = 3  # Default number of neighbors
        
    def create_fruit_dataset(self):
        """Create a custom fruit dataset"""
        
        # Fruit data: [weight, sweetness, color, size]
        self.fruit_data = np.array([
            # Apples (Red) - class 0
            [150, 7, 1, 3],   # Red Apple
            [155, 8, 1, 3],   # Red Apple
            [145, 6, 1, 2],   # Red Apple
            [160, 7, 1, 4],   # Red Apple
            [148, 8, 1, 3],   # Red Apple
            [152, 7, 1, 3],   # Red Apple
            
            # Apples (Green) - class 1
            [140, 6, 2, 2],   # Green Apple
            [145, 7, 2, 3],   # Green Apple
            [135, 5, 2, 2],   # Green Apple
            [150, 6, 2, 3],   # Green Apple
            [142, 7, 2, 2],   # Green Apple
            [138, 6, 2, 2],   # Green Apple
            
            # Oranges - class 2
            [180, 9, 3, 4],   # Orange
            [175, 8, 3, 4],   # Orange
            [185, 9, 3, 4],   # Orange
            [170, 8, 3, 3],   # Orange
            [178, 9, 3, 4],   # Orange
            [182, 8, 3, 4],   # Orange
            
            # Bananas - class 3
            [120, 8, 3, 1],   # Banana
            [125, 9, 3, 1],   # Banana
            [115, 7, 3, 1],   # Banana
            [130, 8, 3, 2],   # Banana
            [118, 8, 3, 1],   # Banana
            [122, 9, 3, 1],   # Banana
            
            # Strawberries - class 4
            [25, 9, 1, 1],    # Strawberry
            [28, 9, 1, 1],    # Strawberry
            [22, 8, 1, 1],    # Strawberry
            [30, 9, 1, 1],    # Strawberry
            [26, 8, 1, 1],    # Strawberry
            [24, 9, 1, 1],    # Strawberry
        ])
        
        # Fruit labels (target values)
        self.fruit_labels = np.array([
            0, 0, 0, 0, 0, 0,      # 6 Red Apples
            1, 1, 1, 1, 1, 1,      # 6 Green Apples
            2, 2, 2, 2, 2, 2,      # 6 Oranges
            3, 3, 3, 3, 3, 3,      # 6 Bananas
            4, 4, 4, 4, 4, 4       # 6 Strawberries
        ])
        
        # Fruit names mapping
        self.fruit_names = {
            0: "Red Apple",
            1: "Green Apple",
            2: "Orange",
            3: "Banana",
            4: "Strawberry"
        }
        
        # Feature names
        self.feature_names = ["Weight (g)", "Sweetness (1-10)", "Color", "Size (1-5)"]
        
        # Color mapping for visualization
        self.color_map = {0: 'red', 1: 'green', 2: 'orange', 3: 'yellow', 4: 'pink'}
        
        print(f"\n📊 Created dataset with {len(self.fruit_data)} fruits")
        print(f"🍎 Fruit types: {len(self.fruit_names)} different kinds")
    
    def explore_dataset(self):
        """Explore and understand the fruit dataset"""
        print("\n" + "="*60)
        print("📊 FRUIT DATASET EXPLORATION")
        print("="*60)
        
        # Dataset overview
        print(f"\nTotal fruits: {len(self.fruit_data)}")
        print(f"Features per fruit: {len(self.feature_names)}")
        print(f"Fruit categories: {len(self.fruit_names)}")
        
        # Show fruit types and counts
        print("\n🍎 Fruit Types in Dataset:")
        print("-" * 40)
        fruit_counts = Counter(self.fruit_labels)
        for fruit_id, count in fruit_counts.items():
            print(f"   {self.fruit_names[fruit_id]}: {count} samples")
        
        # Show feature ranges
        print("\n📏 Feature Ranges:")
        print("-" * 40)
        for i, feature in enumerate(self.feature_names):
            min_val = np.min(self.fruit_data[:, i])
            max_val = np.max(self.fruit_data[:, i])
            mean_val = np.mean(self.fruit_data[:, i])
            print(f"   {feature}:")
            print(f"      Min: {min_val:.1f}")
            print(f"      Max: {max_val:.1f}")
            print(f"      Average: {mean_val:.1f}")
    
    def show_sample_fruits(self, num_samples=5):
        """Display sample fruits from the dataset"""
        print("\n" + "="*60)
        print("🍎 SAMPLE FRUITS FROM DATASET")
        print("="*60)
        
        # Create a table header
        print(f"\n{'No.':<4} {'Fruit':<15} {'Weight':<10} {'Sweetness':<12} {'Color':<10} {'Size':<8}")
        print("-" * 60)
        
        # Show random samples
        indices = np.random.choice(len(self.fruit_data), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            fruit = self.fruit_data[idx]
            fruit_type = self.fruit_names[self.fruit_labels[idx]]
            
            # Convert color code to name
            color_names = {1: "Red", 2: "Green", 3: "Yellow"}
            color = color_names.get(int(fruit[2]), "Unknown")
            
            print(f"{i+1:<4} {fruit_type:<15} {fruit[0]:<10.1f} {fruit[1]:<12} {color:<10} {int(fruit[3]):<8}")
    
    def prepare_data(self, test_size=0.3):
        """Prepare data for training and testing"""
        print("\n" + "="*60)
        print("🔄 PREPARING DATA FOR TRAINING")
        print("="*60)
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.fruit_data, self.fruit_labels, test_size=test_size, 
            random_state=42, stratify=self.fruit_labels
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training set: {len(self.X_train)} fruits ({len(self.X_train)/len(self.fruit_data)*100:.0f}%)")
        print(f"   Testing set: {len(self.X_test)} fruits ({len(self.X_test)/len(self.fruit_data)*100:.0f}%)")
        
        # Show distribution in training set
        print(f"\n📈 Training Set Distribution:")
        train_counts = Counter(self.y_train)
        for fruit_id in sorted(train_counts.keys()):
            percentage = train_counts[fruit_id] / len(self.X_train) * 100
            print(f"   {self.fruit_names[fruit_id]}: {train_counts[fruit_id]} ({percentage:.0f}%)")
    
    def train_model(self, k=None):
        """Train the KNN model"""
        if k:
            self.k_value = k
        
        print("\n" + "="*60)
        print(f"🤖 TRAINING KNN MODEL (k={self.k_value})")
        print("="*60)
        
        # Create and train KNN model
        self.model = KNeighborsClassifier(n_neighbors=self.k_value)
        self.model.fit(self.X_train, self.y_train)
        
        print(f"\n✅ Model training complete!")
        print(f"   K-Nearest Neighbors (KNN) finds the {self.k_value} most similar fruits")
        print(f"   and predicts based on majority vote")
    
    def explain_knn(self):
        """Explain how KNN works"""
        print("\n" + "="*60)
        print("📚 HOW K-NEAREST NEIGHBORS (KNN) WORKS")
        print("="*60)
        
        print("""
KNN is like asking your friends for advice:

1. **Choose K** (number of friends to ask) - We use k=3
2. **Find similarities** - Look for fruits most similar to the new one
3. **Ask neighbors** - Check what type those similar fruits are
4. **Majority vote** - The most common type among neighbors wins!

Example:
   If you have a new fruit:
   - Its 3 most similar fruits are: Apple, Apple, Orange
   - Prediction: APPLE (because 2 out of 3 neighbors are apples)

Key Concept: "Birds of a feather flock together!"
        """)
        
        # Visual representation
        self.draw_knn_explanation()
    
    def draw_knn_explanation(self):
        """Draw a simple diagram explaining KNN"""
        plt.figure(figsize=(10, 6))
        
        # Create sample points for demonstration
        np.random.seed(42)
        
        # Three different fruit types
        apples = np.random.randn(10, 2) * 0.5 + np.array([2, 2])
        oranges = np.random.randn(10, 2) * 0.5 + np.array([4, 4])
        bananas = np.random.randn(10, 2) * 0.5 + np.array([3, 1])
        
        # New fruit to classify
        new_fruit = np.array([3, 2.5])
        
        # Plot existing fruits
        plt.scatter(apples[:, 0], apples[:, 1], c='red', label='Apples', s=100, alpha=0.6)
        plt.scatter(oranges[:, 0], oranges[:, 1], c='orange', label='Oranges', s=100, alpha=0.6)
        plt.scatter(bananas[:, 0], bananas[:, 1], c='yellow', label='Bananas', s=100, alpha=0.6)
        
        # Plot new fruit
        plt.scatter(new_fruit[0], new_fruit[1], c='blue', marker='*', 
                   s=500, label='New Fruit?', edgecolors='black', linewidth=2)
        
        # Find and highlight 3 nearest neighbors
        all_points = np.vstack([apples, oranges, bananas])
        distances = np.sqrt(np.sum((all_points - new_fruit)**2, axis=1))
        nearest_indices = np.argsort(distances)[:3]
        
        # Draw circles around nearest neighbors
        for idx in nearest_indices:
            circle = plt.Circle(all_points[idx], 0.3, fill=False, color='green', linewidth=2)
            plt.gca().add_patch(circle)
        
        plt.title('How KNN Works: Find the 3 Nearest Neighbors', fontsize=14)
        plt.xlabel('Feature 1 (e.g., Weight)', fontsize=12)
        plt.ylabel('Feature 2 (e.g., Sweetness)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Add explanation text
        plt.text(0.5, -0.1, 
                'The new fruit (★) looks at its 3 nearest neighbors (circled)\n'
                'Majority vote determines the fruit type!',
                transform=plt.gca().transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self):
        """Evaluate the model's performance"""
        if not self.model:
            print("❌ Please train the model first!")
            return
        
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION")
        print("="*60)
        
        # Make predictions on test set
        y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\n✅ Overall Accuracy: {accuracy:.2%}")
        
        # Detailed per-fruit performance
        print(f"\n📈 Performance by Fruit Type:")
        print("-" * 50)
        
        for fruit_id in sorted(self.fruit_names.keys()):
            # Find indices where true fruit is this type
            fruit_indices = np.where(self.y_test == fruit_id)[0]
            if len(fruit_indices) > 0:
                correct_predictions = np.sum(y_pred[fruit_indices] == fruit_id)
                total = len(fruit_indices)
                fruit_accuracy = correct_predictions / total
                
                # Show bar for visual representation
                bar = '█' * int(fruit_accuracy * 20)
                print(f"   {self.fruit_names[fruit_id]:<15}: {bar:20} {fruit_accuracy:.0%} ({correct_predictions}/{total})")
        
        # Show some example predictions
        print(f"\n🔍 Sample Predictions:")
        print("-" * 50)
        print(f"{'Actual':<15} {'Predicted':<15} {'Correct?'}")
        print("-" * 50)
        
        # Show 5 random test samples
        sample_indices = np.random.choice(len(self.X_test), 5, replace=False)
        for idx in sample_indices:
            actual = self.fruit_names[self.y_test[idx]]
            predicted = self.fruit_names[y_pred[idx]]
            correct = "✅" if actual == predicted else "❌"
            print(f"{actual:<15} {predicted:<15} {correct}")
    
    def find_optimal_k(self, max_k=10):
        """Find the best K value"""
        print("\n" + "="*60)
        print("🔍 FINDING OPTIMAL K VALUE")
        print("="*60)
        
        k_values = range(1, max_k + 1)
        accuracies = []
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.y_train)
            y_pred = knn.predict(self.X_test)
            accuracies.append(accuracy_score(self.y_test, y_pred))
        
        # Find best k
        best_k = k_values[np.argmax(accuracies)]
        best_accuracy = max(accuracies)
        
        print(f"\n📊 Results:")
        for k, acc in zip(k_values, accuracies):
            star = "★" if k == best_k else " "
            print(f"   {star} k={k}: Accuracy = {acc:.2%}")
        
        print(f"\n✅ Best k value: {best_k} (Accuracy: {best_accuracy:.2%})")
        
        # Plot k values vs accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.plot(best_k, best_accuracy, 'r*', markersize=20, label=f'Best k={best_k}')
        plt.xlabel('K Value (Number of Neighbors)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Finding the Best K Value for KNN', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        plt.legend()
        
        # Add explanation
        plt.text(0.5, -0.15, 
                'Small k: More sensitive to noise\n'
                'Large k: Smoother but might miss patterns',
                transform=plt.gca().transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        return best_k
    
    def visualize_fruits(self):
        """Create visualizations of the fruit dataset"""
        print("\n" + "="*60)
        print("🎨 CREATING FRUIT VISUALIZATIONS")
        print("="*60)
        
        # Create a 2x2 grid of plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fruit Dataset Visualization', fontsize=16, y=1.02)
        
        # Plot 1: Weight vs Sweetness
        ax = axes[0, 0]
        for fruit_id, color in self.color_map.items():
            mask = self.fruit_labels == fruit_id
            ax.scatter(self.fruit_data[mask, 0], self.fruit_data[mask, 1], 
                      c=color, label=self.fruit_names[fruit_id], s=100, alpha=0.7)
        ax.set_xlabel('Weight (grams)', fontsize=11)
        ax.set_ylabel('Sweetness (1-10)', fontsize=11)
        ax.set_title('Weight vs Sweetness', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Size vs Color
        ax = axes[0, 1]
        for fruit_id, color in self.color_map.items():
            mask = self.fruit_labels == fruit_id
            # Jitter the color values slightly for better visualization
            size_data = self.fruit_data[mask, 3] + np.random.normal(0, 0.05, np.sum(mask))
            color_data = self.fruit_data[mask, 2] + np.random.normal(0, 0.05, np.sum(mask))
            ax.scatter(size_data, color_data, c=color, 
                      label=self.fruit_names[fruit_id], s=100, alpha=0.7)
        ax.set_xlabel('Size (1-5)', fontsize=11)
        ax.set_ylabel('Color Code (1-3)', fontsize=11)
        ax.set_title('Size vs Color', fontsize=12)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Red', 'Green', 'Yellow'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: 3D-like visualization (using bubble size for weight)
        ax = axes[1, 0]
        for fruit_id, color in self.color_map.items():
            mask = self.fruit_labels == fruit_id
            # Use sweetness for x, size for y, and weight for bubble size
            sizes = self.fruit_data[mask, 0] / 5  # Scale down for bubble size
            ax.scatter(self.fruit_data[mask, 1], self.fruit_data[mask, 3], 
                      s=sizes, c=color, alpha=0.6, label=self.fruit_names[fruit_id])
        ax.set_xlabel('Sweetness', fontsize=11)
        ax.set_ylabel('Size', fontsize=11)
        ax.set_title('Bubble Chart (bubble size = weight)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Feature importance visualization
        ax = axes[1, 1]
        feature_importance = self.calculate_feature_importance()
        y_pos = np.arange(len(self.feature_names))
        ax.barh(y_pos, feature_importance, color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.feature_names)
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_title('Feature Importance in Classification', fontsize=12)
        ax.invert_yaxis()  # Display highest importance at top
        
        # Add value labels
        for i, v in enumerate(feature_importance):
            ax.text(v + 0.01, i, f'{v:.2f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_feature_importance(self):
        """Calculate rough feature importance based on variance between classes"""
        importance = []
        
        for feature_idx in range(len(self.feature_names)):
            # Calculate between-class variance / within-class variance
            feature_data = self.fruit_data[:, feature_idx]
            overall_mean = np.mean(feature_data)
            
            between_class_var = 0
            within_class_var = 0
            
            for fruit_id in range(len(self.fruit_names)):
                class_data = feature_data[self.fruit_labels == fruit_id]
                class_mean = np.mean(class_data)
                class_var = np.var(class_data)
                
                between_class_var += len(class_data) * (class_mean - overall_mean) ** 2
                within_class_var += len(class_data) * class_var
            
            # F-ratio (higher means better separation)
            if within_class_var > 0:
                f_ratio = between_class_var / within_class_var
                importance.append(f_ratio)
            else:
                importance.append(0)
        
        # Normalize to 0-1 range
        importance = np.array(importance)
        if np.max(importance) > 0:
            importance = importance / np.max(importance)
        
        return importance
    
    def predict_new_fruit(self):
        """Predict fruit type for a new fruit"""
        print("\n" + "="*60)
        print("🔮 PREDICT A NEW FRUIT")
        print("="*60)
        print("\nEnter the characteristics of your mystery fruit:")
        
        try:
            # Get fruit measurements
            weight = float(input("   Weight (grams): "))
            sweetness = float(input("   Sweetness (1-10): "))
            
            print("\n   Color options:")
            print("      1. Red")
            print("      2. Green")
            print("      3. Yellow")
            color = int(input("   Enter color code (1-3): "))
            
            size = float(input("   Size (1-5, where 1=small, 5=large): "))
            
            # Create feature array
            new_fruit = np.array([[weight, sweetness, color, size]])
            
            # Make prediction
            prediction = self.model.predict(new_fruit)[0]
            probabilities = self.model.predict_proba(new_fruit)[0]
            
            # Find nearest neighbors
            distances, indices = self.model.kneighbors(new_fruit)
            
            print(f"\n🎯 Prediction Result:")
            print(f"   This fruit is likely a: {self.fruit_names[prediction].upper()}!")
            
            print(f"\n   Confidence Levels:")
            # Sort probabilities
            fruit_probs = [(self.fruit_names[i], prob) for i, prob in enumerate(probabilities)]
            fruit_probs.sort(key=lambda x: x[1], reverse=True)
            
            for fruit_name, prob in fruit_probs[:3]:  # Show top 3
                if prob > 0:
                    bar = '█' * int(prob * 20)
                    print(f"   {fruit_name:<15}: {bar:20} {prob:.1%}")
            
            print(f"\n   🤔 How did we decide?")
            print(f"   Looked at the {self.k_value} most similar fruits:")
            
            for i, idx in enumerate(indices[0]):
                neighbor_fruit = self.X_train[idx]
                neighbor_type = self.fruit_names[self.y_train[idx]]
                distance = distances[0][i]
                
                print(f"\n   Neighbor {i+1}: {neighbor_type}")
                print(f"      Weight: {neighbor_fruit[0]:.0f}g (yours: {weight:.0f}g)")
                print(f"      Sweetness: {neighbor_fruit[1]:.0f}/10 (yours: {sweetness:.0f}/10)")
                print(f"      Similarity score: {1/(1+distance):.2f}")
            
        except ValueError:
            print("❌ Invalid input! Please enter numbers only.")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def interactive_fruit_game(self):
        """Play an interactive fruit guessing game"""
        print("\n" + "="*60)
        print("🎮 FRUIT GUESSING GAME")
        print("="*60)
        print("\nThink of a fruit, and I'll try to guess it!")
        print("Answer these questions about your fruit:")
        
        questions = [
            "Is it heavy? (100g is about the weight of a small apple)",
            "Is it very sweet?",
            "Is it red in color?",
            "Is it small in size? (like a strawberry)"
        ]
        
        # Get user input
        answers = []
        for q in questions:
            answer = input(f"\n{q} (yes/no): ").lower().strip()
            answers.append(answer == 'yes' or answer == 'y')
        
        # Convert answers to features (simplified mapping)
        weight = 150 if answers[0] else 50  # Heavy or light
        sweetness = 8 if answers[1] else 5  # Sweet or not
        color = 1 if answers[2] else 2      # Red or not
        size = 1 if answers[3] else 3       # Small or not
        
        # Create feature array
        guess_fruit = np.array([[weight, sweetness, color, size]])
        
        # Predict
        prediction = self.model.predict(guess_fruit)[0]
        
        print(f"\n🎯 My guess: You're thinking of a {self.fruit_names[prediction]}!")
        
        # Show probabilities
        probs = self.model.predict_proba(guess_fruit)[0]
        print("\nOther possibilities:")
        for i, prob in enumerate(probs):
            if prob > 0.1 and i != prediction:
                print(f"   Could also be {self.fruit_names[i]} ({prob:.0%} confidence)")
    
    def interactive_menu(self):
        """Interactive menu for user"""
        while True:
            print("\n" + "="*60)
            print("🍎 INTERACTIVE MENU")
            print("="*60)
            print("1. Show dataset information")
            print("2. Show sample fruits")
            print("3. Train KNN model")
            print("4. Explain how KNN works")
            print("5. Find optimal K value")
            print("6. Evaluate model performance")
            print("7. Visualize fruits")
            print("8. Predict a new fruit")
            print("9. Play fruit guessing game")
            print("10. Exit")
            
            choice = input("\nEnter your choice (1-10): ")
            
            if choice == '1':
                self.explore_dataset()
            elif choice == '2':
                self.show_sample_fruits(8)
            elif choice == '3':
                k = input(f"Enter K value (press Enter for default={self.k_value}): ")
                if k.strip():
                    self.train_model(int(k))
                else:
                    self.train_model()
            elif choice == '4':
                self.explain_knn()
            elif choice == '5':
                best_k = self.find_optimal_k(15)
                use_best = input(f"\nUse best k={best_k} for training? (yes/no): ")
                if use_best.lower() in ['yes', 'y']:
                    self.train_model(best_k)
            elif choice == '6':
                self.evaluate_model()
            elif choice == '7':
                self.visualize_fruits()
            elif choice == '8':
                if self.model:
                    self.predict_new_fruit()
                else:
                    print("⚠️  Please train the model first (option 3)!")
            elif choice == '9':
                if self.model:
                    self.interactive_fruit_game()
                else:
                    print("⚠️  Please train the model first (option 3)!")
            elif choice == '10':
                print("\n🍎 Thanks for playing with the Fruit Classifier! Goodbye! 🍎")
                break
            else:
                print("❌ Invalid choice! Please try again.")

def main():
    """Main function to run the Fruit Classifier"""
    
    # Create classifier
    classifier = FruitClassifier()
    
    # Step 1: Explore the dataset
    classifier.explore_dataset()
    classifier.show_sample_fruits()
    
    # Step 2: Prepare data
    classifier.prepare_data()
    
    # Step 3: Explain KNN
    classifier.explain_knn()
    
    # Step 4: Find best K
    input("\nPress Enter to find the best K value...")
    best_k = classifier.find_optimal_k()
    
    # Step 5: Train with best K
    classifier.train_model(best_k)
    
    # Step 6: Evaluate
    classifier.evaluate_model()
    
    # Step 7: Visualize
    classifier.visualize_fruits()
    
    # Step 8: Make some predictions
    print("\n" + "🎯" * 30)
    print("     TIME TO TEST THE MODEL!")
    print("🎯" * 30)
    
    # Test with some sample fruits
    test_samples = [
        [155, 7, 1, 3],  # Should be Red Apple
        [140, 6, 2, 2],  # Should be Green Apple
        [180, 9, 3, 4],  # Should be Orange
        [120, 8, 3, 1],  # Should be Banana
        [25, 9, 1, 1],   # Should be Strawberry
    ]
    
    print("\n📝 Testing known fruits:")
    for i, sample in enumerate(test_samples):
        pred = classifier.model.predict([sample])[0]
        print(f"\n   Test {i+1}:")
        print(f"      Weight: {sample[0]}g, Sweetness: {sample[1]}, Size: {sample[3]}")
        print(f"      Predicted: {classifier.fruit_names[pred]}")
    
    # Interactive prediction
    print("\n" + "✨" * 30)
    print("     NOW IT'S YOUR TURN!")
    print("✨" * 30)
    classifier.predict_new_fruit()
    
    # Interactive menu for further exploration
    classifier.interactive_menu()

# Run the program
if __name__ == "__main__":
    main()