import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter as Router, 
  Switch, 
  Route, 
  Link, 
} from 'react-router-dom'
//React Router von: https://reactrouter.com
import './index.css';
import Results from './components/pages/Results.js';
import AlgorithmSelector from './components/pages/AlgorithmSelector.js';
import LandingPage from './components/pages/LandingPage.js';
import ActiveContainerList from './components/pages/ContainerList'; 
import Header from './Header.js';
import Particles from 'react-particles-js';

//Rendert den Homescreen auf der Benutzeroberfläche,
//von dem aus man die Komponente DynamicResults oder AlgorithmSelector laden kann
export default function App(){
  return(
    <Router>
      <Header />
      <div className="navbar-offset-container">
        <div className="grid-container">
          <Particles 
            params={{
              "particles": {
                "number": {
                  "value": 80
                },
                "move": {
                  "speed": 0.5
                },
                "color": {
                  "value": "#0c3f93"
                },
                "line_linked": {
                  color: '#bbb',
                  opacity: 1
                }
              }
            }}
              style={{position:"fixed", zIndex:-1, height:"100vh"}}
            />
          <div className="container">
              <Switch>
                <Route exact path="/" children={<LandingPage/>}/>
                <Route path="/new_calc" children={<AlgorithmSelector/>}/>
                <Route path="/results" children={<Results/>}/>
                <Route path="/active_containers" children={<ActiveContainerList/>}/>
              </Switch>
          </div>
        </div>
      </div>
    </Router>
  );
}


ReactDOM.render(<App/>, document.getElementById('root'));
