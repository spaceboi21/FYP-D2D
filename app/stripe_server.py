# stripe_server.py

from flask import Flask, request, jsonify
import stripe
import os
from dotenv import load_dotenv
from cors_config import configure_cors

load_dotenv()

app = Flask(__name__)
app = configure_cors(app)  # Configure CORS with our settings

# ✅ Set your Stripe secret key
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")  

@app.route('/')
def home():
    return 'Stripe Payment Server is running!'

@app.route('/pay', methods=['POST'])
def pay():
    try:
        data = request.get_json()
        print(f"DEBUG: /pay received payload: {data}")

        payment_method_id = data.get('payment_method')
        amount = data.get('amount', 0)

        if not payment_method_id or not amount:
            return jsonify({'success': False, 'error': 'Missing payment_method or amount'}), 400

        # Create the PaymentIntent
        intent = stripe.PaymentIntent.create(
            amount=amount,  # in cents
            currency='usd',
            payment_method=payment_method_id,
            confirm=True,
            automatic_payment_methods={'enabled': True,
                                       'allow_redirects': 'never'}
        )

        return jsonify({'success': True, 'message': 'Payment successful'})

    except stripe.error.CardError as e:
        return jsonify({'success': False, 'error': str(e.user_message)}), 402

    except Exception as e:
        print(f"DEBUG: Error during payment: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5050, debug=True)
