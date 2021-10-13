import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter as Router, 
  Switch, 
  Route, 
  Link, 
} from 'react-router-dom'
//React Router von: https://reactrouter.com
import './index.css';
import DynamicResults from './dynamic_results.js';
import AlgorithmSelector from './dynamic_form.js';
import logo from "./hhu_logo.png";

//Rendert den Homescreen auf der Benutzeroberfläche,
//von dem aus man die Komponente DynamicResults oder AlgorithmSelector laden kann
export default function App(){
  return(
    <Router>
      <div className="container">
        <div className="header">
          <div className="logo">
            <img src={logo} alt="HHU Logo"/>
          </div>
          <div className="navbar">
            <ul>
              <li>
                <Link to="/">Start New Calculation</Link>
              </li>
              <li>
                <Link to="/results">Results</Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="content">
          <Switch>
            <Route path="/results" children={<DynamicResults/>}/>
            <Route path="/" children={<AlgorithmSelector/>}/>
          </Switch>
        </div>

        <div className="footer">
        </div>
      </div>
    </Router>
  );
}


ReactDOM.render(<App/>, document.getElementById('root'));
