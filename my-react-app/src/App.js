import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Hero from "./components/Hero";
import Navbar from "./components/Navbar";
import { Analytics } from "./components/Analytics";
import Newsletter from "./components/Newsletter";
import Cards from "./components/Cards";
import Footer from "./components/Footer";
import CheckoutForm from "./components/CheckoutForm";
import Demo from "./components/demo";
import Team from "./components/Team";
import { Elements } from "@stripe/react-stripe-js";
import { loadStripe } from "@stripe/stripe-js";

// 👇 Replace this with your own public test key from Stripe Dashboard
const stripePromise = loadStripe("pk_test_51RD38jI5powDjew556mcuzZ4MJ0yP1w6Ut7q67FoDvKwKsLVnYxct7uruvYw54wia8itYjciCmaDur1s9SntzsQi00SUAWy1nk");

const Home = () => (
  <div>
    <Navbar />
    <div id="home"><Hero /></div>
    <div id="dashboard"><Analytics /></div>
    <div id="guide"><Newsletter /></div>
    <div id="demo"><Demo /></div>
    <div id="team"><Team /></div>
    <div id="pricing"><Cards /></div>
    <div id="resources"><Footer /></div>
  </div>
);

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route
          path="/checkout/:plan"
          element={
            <Elements stripe={stripePromise}>
              <CheckoutForm />
            </Elements>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
