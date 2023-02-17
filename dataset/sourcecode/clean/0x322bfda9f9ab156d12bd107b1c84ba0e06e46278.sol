pragma solidity ^0.4.0;

contract SimpleStorage
{
    uint storedData;
    
    function set(uint x) public
    {
        storedData = x;
    }

    function get() public View returns (uint)
    {
        return storedData;
    }
}