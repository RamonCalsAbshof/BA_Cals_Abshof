import React from 'react';
import styled from 'styled-components';
import logo from "./hhu_logo.png";
import { Link } from 'react-router-dom';

const NavBarWrapper = styled.div`
  padding: 10px 50px;
  display: flex;
  height: 50px;
  box-shadow: rgba(0, 0, 0, 0.14) 0px 4px 5px 0px;
  background-color: white; 
  width: 100%;
  position:fixed;
  z-index:1;
`;

const NavBar = styled.header`
  display: flex;
  flex: 0.35;
  align-items: center;
`;

const NavItem = styled(Link)`
  font-size: 14px;
  padding: 10px 20px 0px 20px;
  color: rgba(0, 0, 0, 0.7);
  text-decoration: none;
  overflow: hidden;
  white-space: nowrap;
`;

const Logo = styled.span`
  color: rgb(10, 84, 164);
  font-size: 45px;
  font-family: Verdana, Geneva, sans-serif;
  font-weight: 600;
  padding-right:30px;
`;

const Header = props => {
  return (
    <NavBarWrapper>
      <NavBar>
        <Logo>hhu</Logo>
        <NavItem to="/">Home</NavItem>
        <NavItem to="/new_calc">Analyse Starten</NavItem>
        <NavItem to="/results">Ergebnisse</NavItem>
      </NavBar>
    </NavBarWrapper>
  )
}

export default Header;