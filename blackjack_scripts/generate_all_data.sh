echo "-------------------------------------------------"
echo "Blackjack Data Generation script by Anthony Grego"
echo "-------------------------------------------------"
echo ""

# Get inside the scripts directory
cd "$(dirname "$0")"

#Development Data
echo "Generating Development Dataset (10,000 hands)..."
python3 blackjack_simulator.py \
  --output_path ../data/development/blackjack_dev_10k.csv \
  --log_path ../data/development/blackjack_dev.log \
  --hands 10000 \
  --decks 8 \
  --pen 6.5
echo "Development data complete"
echo ""

#Training
echo "Generating Training Dataset (100,000 hands)..."
python3 blackjack_simulator.py \
  --output_path ../data/training/blackjack_train_100k.csv \
  --log_path ../data/training/blackjack_train.log \
  --hands 100000 \
  --decks 8 \
  --pen 6.5
echo "Training data complete."
echo ""

#Validation
echo "Generating Validation Dataset (20,000 hands)..."
python3 blackjack_simulator.py \
  --output_path ../data/validation/blackjack_val_20k.csv \
  --log_path ../data/validation/blackjack_val.log \
  --hands 20000 \
  --decks 8 \
  --pen 6.5
echo "Validation data complete."
echo ""

#Summary
echo "-----------------------------------------------"
echo "All datasets generated successfully!"
echo "-----------------------------------------------"
echo ""
echo "Datasets:"
echo "  - Development: data/development/blackjack_dev_10k.csv (10,000 hands)"
echo "  - Training: data/training/blackjack_train_100k.csv (100,000 hands)"
echo "  - Validation: data/validation/blackjack_val_20k.csv (20,000 hands)"
echo ""
echo "Total # of hands generated: 130,000"
